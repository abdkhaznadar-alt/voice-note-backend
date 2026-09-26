import asyncio
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicenote-backend")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
# Stock faster-whisper model size, used as a fallback if the dialect-tuned
# CTranslate2 model was not baked into the image (e.g. local dev without
# running the Docker build step).
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
MAX_TRANSCRIBE_WORKERS = 2

# Free, open-source Whisper model fine-tuned specifically on Arabic dialects
# (Gulf / Levantine / Egyptian / Iraqi / etc.), converted to CTranslate2
# format at Docker build time (see Dockerfile). Using this instead of the
# stock "small" model noticeably improves dialect transcription quality at
# zero extra cost (no external API calls involved).
DIALECT_MODEL_REPO = "oddadmix/whisper-small-arabic-dialectal-v2"
DIALECT_MODEL_CT2_DIR = os.environ.get(
    "DIALECT_MODEL_CT2_DIR", "/app/models/whisper-small-arabic-dialectal-v2-ct2"
)
USE_DIALECT_MODEL = os.environ.get("USE_DIALECT_MODEL", "true").lower() == "true"

_model_path = MODEL_SIZE
_model_source = "stock"
if USE_DIALECT_MODEL and os.path.isdir(DIALECT_MODEL_CT2_DIR) and os.listdir(DIALECT_MODEL_CT2_DIR):
    _model_path = DIALECT_MODEL_CT2_DIR
    _model_source = "dialect-tuned"
else:
    logger.warning(
        "Dialect-tuned model not found at %s (or USE_DIALECT_MODEL=false); "
        "falling back to stock Whisper '%s'.",
        DIALECT_MODEL_CT2_DIR,
        MODEL_SIZE,
    )

logger.info("Loading Whisper model: %s (source=%s)", _model_path, _model_source)
whisper_model = WhisperModel(_model_path, device="cpu", compute_type=COMPUTE_TYPE)
logger.info("Whisper model loaded.")

# ---------------------------------------------------------------------------
# Gemini configuration (server-side only — never shipped in the app)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"

app = FastAPI(title="VoiceNote AI Backend")

executor = ThreadPoolExecutor(max_workers=MAX_TRANSCRIBE_WORKERS)

jobs: dict = {}
jobs_lock = threading.Lock()


class JobCanceled(Exception):
    pass


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------
def _gemini_url(json_mode: bool) -> str:
    return f"{GEMINI_BASE_URL}/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"


def _gemini_payload(prompt: str, json_mode: bool) -> dict:
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    if json_mode:
        payload["generationConfig"] = {"response_mime_type": "application/json"}
    return payload


def _extract_gemini_text(data: dict) -> str:
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Unexpected Gemini response shape: {data}")


async def call_gemini(prompt: str, json_mode: bool = False) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured on the server.")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(_gemini_url(json_mode), json=_gemini_payload(prompt, json_mode))
        resp.raise_for_status()
        return _extract_gemini_text(resp.json())


def _call_gemini_sync(prompt: str, json_mode: bool = False) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured on the server.")
    resp = requests.post(_gemini_url(json_mode), json=_gemini_payload(prompt, json_mode), timeout=60.0)
    resp.raise_for_status()
    return _extract_gemini_text(resp.json())


def _build_analyze_prompt(text: str, language: str) -> str:
    return f"""You are an assistant that extracts structured information from a voice note transcript.
Language of the transcript: {language}.
Return ONLY valid JSON with these keys: summary (string), key_points (array of strings),
tasks (array of strings), deadlines (array of strings), dates (array of strings),
times (array of strings), people (array of strings), decisions (array of strings),
questions (array of strings). If a category has nothing, return an empty array.

Transcript:
\"\"\"{text}\"\"\"
"""


# ---------------------------------------------------------------------------
# Transcription core
# ---------------------------------------------------------------------------
def _transcribe_file_sync(tmp_path: str, forced_language: Optional[str], job_id: Optional[str] = None) -> str:
    # An empty string (sent by some clients to mean "auto-detect") is not a
    # valid Whisper language code — only None triggers auto-detection.
    if not forced_language:
        forced_language = None

    segments, info = whisper_model.transcribe(
        tmp_path,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        language=forced_language,
    )

    total_duration = getattr(info, "duration", None) or 0.0
    texts = []

    for segment in segments:
        if job_id is not None:
            with jobs_lock:
                job = jobs.get(job_id)
                if job is None:
                    raise JobCanceled("Job no longer exists")
                if job.get("canceled"):
                    raise JobCanceled("Job canceled by user")

        # Drop segments that look like hallucinations on silence / noise.
        if segment.no_speech_prob > 0.6 or segment.avg_logprob < -1.0:
            continue

        texts.append(segment.text.strip())

        if job_id is not None and total_duration > 0:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["progress"] = min(segment.end / total_duration, 0.99)

    return " ".join(t for t in texts if t)


def _run_job(job_id: str, tmp_path: str, forced_language: Optional[str], also_analyze: bool = False):
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "running"

        text = _transcribe_file_sync(tmp_path, forced_language, job_id=job_id)

        result = {"text": text}

        if also_analyze and text.strip():
            try:
                language_label = forced_language or "auto"
                prompt = _build_analyze_prompt(text, language_label)
                analysis_raw = _call_gemini_sync(prompt, json_mode=True)
                result["analysis"] = json.loads(analysis_raw)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Combined analysis failed for job %s", job_id)
                result["analysis"] = None
                result["analysis_error"] = str(exc)

        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["result"] = result

    except JobCanceled:
        with jobs_lock:
            jobs[job_id]["status"] = "canceled"

    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(exc)

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _cleanup_old_jobs(max_age_seconds: int = 3600):
    now = time.time()
    with jobs_lock:
        stale = [
            jid
            for jid, job in jobs.items()
            if job.get("status") in ("done", "error", "canceled")
            and now - job.get("finished_at", job.get("created_at", now)) > max_age_seconds
        ]
        for jid in stale:
            jobs.pop(jid, None)
    if stale:
        logger.info("Cleaned up %d stale job(s).", len(stale))


async def _cleanup_loop():
    while True:
        await asyncio.sleep(600)
        _cleanup_old_jobs()


@app.on_event("startup")
async def _on_startup():
    asyncio.create_task(_cleanup_loop())


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    target_language: str


class AnalyzeRequest(BaseModel):
    text: str
    language: str = "auto"


class AskRequest(BaseModel):
    transcript: str
    question: str
    language: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    with jobs_lock:
        active = sum(1 for j in jobs.values() if j.get("status") in ("queued", "running"))
    return {
        "status": "ok",
        "whisper_model": _model_path,
        "whisper_model_source": _model_source,
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "max_transcribe_workers": MAX_TRANSCRIBE_WORKERS,
        "active_jobs": active,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: Optional[str] = Form(None)):
    suffix = os.path.splitext(file.filename or "")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(executor, _transcribe_file_sync, tmp_path, language, None)
        return {"text": text}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Legacy /transcribe failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/transcribe/start")
async def transcribe_start(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    also_analyze: bool = Form(False),
):
    _cleanup_old_jobs()

    suffix = os.path.splitext(file.filename or "")[1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "result": None,
            "error": None,
            "canceled": False,
            "created_at": time.time(),
        }

    executor.submit(_run_job, job_id, tmp_path, language, also_analyze)

    return JSONResponse(status_code=202, content={"job_id": job_id})


@app.get("/transcribe/status/{job_id}")
async def transcribe_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in ("done", "error", "canceled") and "finished_at" not in job:
            job["finished_at"] = time.time()
        return {
            "status": job["status"],
            "progress": job.get("progress", 0.0),
            "result": job.get("result"),
            "error": job.get("error"),
        }


@app.post("/transcribe/cancel/{job_id}")
async def transcribe_cancel(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        job["canceled"] = True
    return {"status": "canceling"}


@app.post("/translate")
async def translate(req: TranslateRequest):
    prompt = (
        f"Translate the following text to {req.target_language}. "
        f"Return ONLY the translated text, no explanations.\n\n{req.text}"
    )
    try:
        translated = await call_gemini(prompt)
        return {"translated_text": translated.strip()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Translate failed")
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    prompt = _build_analyze_prompt(req.text, req.language)
    try:
        raw = await call_gemini(prompt, json_mode=True)
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analyze failed")
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/ask")
async def ask(req: AskRequest):
    prompt = (
        f"Given this voice note transcript:\n\"\"\"{req.transcript}\"\"\"\n\n"
        f"Answer this question about it: {req.question}\n"
        f"Answer in {req.language or 'the same language as the transcript'}."
    )
    try:
        answer = await call_gemini(prompt)
        return {"answer": answer.strip()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ask failed")
        raise HTTPException(status_code=502, detail=str(exc))
