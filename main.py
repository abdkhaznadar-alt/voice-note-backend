"""
VoiceNote AI backend — transcription (Faster-Whisper, local) + translation/
analysis (Gemini, called server-side so the API key never ships in the app).

RUN LOCALLY:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

RUN ON HUGGING FACE SPACES (Docker):
    Deployed via the included Dockerfile, which listens on port 7860
    (Spaces' required port). Set GEMINI_API_KEY as a "Secret" in your
    Space's Settings — it's read the exact same way (os.environ) either way.

Set your Gemini key as an environment variable before starting locally (never hardcode it):
    export GEMINI_API_KEY="your-key-here"   # Windows: set GEMINI_API_KEY=your-key-here

QUEUE FOR LONG RECORDINGS:
    Transcription runs through a bounded worker pool (MAX_TRANSCRIBE_WORKERS)
    instead of running unbounded per-request — this is what keeps a handful
    of people uploading long recordings at once from all fighting over the
    same CPU and slowing each other to a crawl. Two ways to use it:

    1) POST /transcribe (unchanged) — still waits for the result on the same
       connection, just now queued behind the worker pool if it's busy.
       Simplest, and what the app already uses.

    2) POST /transcribe/start + GET /transcribe/status/{job_id} — submit a
       job and get a job_id back immediately (no long-held connection),
       then poll for its status. Better for long recordings and flaky
       mobile connections, where holding one HTTP request open for minutes
       is fragile.

Test it with:
    curl -F "file=@some_recording.m4a" http://localhost:8000/transcribe
    curl -F "file=@some_recording.m4a" http://localhost:8000/transcribe/start
    curl http://localhost:8000/transcribe/status/<job_id>
    curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" \
         -d '{"text": "...", "language": "ar"}'
    curl -X POST http://localhost:8000/translate -H "Content-Type: application/json" \
         -d '{"text": "...", "target_language": "German"}'
"""
import asyncio
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicenote-backend")

app = FastAPI(title="VoiceNote AI Backend")

# ---------------------------------------------------------------------------
# Faster-Whisper — loaded once at startup, local, free, no API key needed.
# ---------------------------------------------------------------------------
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"

logger.info(f"Loading Faster-Whisper model '{MODEL_SIZE}' (compute_type={COMPUTE_TYPE})...")
whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
logger.info("Model loaded — server ready.")

# ---------------------------------------------------------------------------
# Transcription queue — a bounded thread pool so N people uploading at once
# can't all spawn simultaneous CPU-bound Whisper runs and grind the machine
# to a halt. Requests beyond MAX_TRANSCRIBE_WORKERS simply wait their turn.
# Tune this to roughly your CPU core count; more than that just adds
# contention rather than real throughput on a CPU-only box.
# ---------------------------------------------------------------------------
MAX_TRANSCRIBE_WORKERS = 2
transcribe_executor = ThreadPoolExecutor(max_workers=MAX_TRANSCRIBE_WORKERS, thread_name_prefix="whisper-worker")

# In-memory job store for the async /transcribe/start + /transcribe/status
# flow. Fine for a single-process server like this one; if you ever run
# multiple server processes/machines behind a load balancer, this would
# need to move to something shared (e.g. Redis) so any process can answer
# a status check regardless of which one is running the job.
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
JOB_RETENTION_SECONDS = 60 * 60  # finished jobs are forgotten after an hour


def _cleanup_old_jobs():
    cutoff = time.time() - JOB_RETENTION_SECONDS
    with jobs_lock:
        stale = [jid for jid, j in jobs.items() if j["status"] in ("done", "failed", "canceled") and j["finished_at"] and j["finished_at"] < cutoff]
        for jid in stale:
            del jobs[jid]
    if stale:
        logger.info(f"Cleaned up {len(stale)} expired job(s)")


async def _periodic_cleanup_task():
    """Runs for the whole life of the server, sweeping expired jobs even
    when no new requests come in to trigger a cleanup on their own."""
    while True:
        await asyncio.sleep(600)  # every 10 minutes
        _cleanup_old_jobs()


@app.on_event("startup")
async def _start_background_tasks():
    asyncio.create_task(_periodic_cleanup_task())


def _transcribe_file_sync(tmp_path: str, forced_language: Optional[str], job_id: Optional[str] = None) -> dict:
    """
    The actual CPU-bound Whisper work — runs inside transcribe_executor's
    worker threads, never on the main event loop. If [job_id] is given
    (the async job flow), updates jobs[job_id]["progress"] as segments
    complete and checks jobs[job_id]["canceled"] between segments so a
    cancel request actually stops further work instead of running to
    completion regardless.
    """
    segments, info = whisper_model.transcribe(
        tmp_path,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        language=forced_language,
    )
    total_duration = info.duration or 0.0
    # Hallucination filter: a segment faster-whisper itself flags as
    # "probably silence" (no_speech_prob) or very low-confidence
    # (avg_logprob) is far more likely to be invented text than a real
    # transcription — drop those rather than including made-up words.
    kept_segments = []
    for segment in segments:
        if job_id is not None:
            with jobs_lock:
                if jobs.get(job_id, {}).get("canceled"):
                    logger.info(f"Job {job_id} canceled mid-transcription — stopping early")
                    raise JobCanceled()
                if total_duration > 0:
                    jobs[job_id]["progress"] = min(0.98, segment.end / total_duration)

        if segment.no_speech_prob > 0.6 or segment.avg_logprob < -1.0:
            logger.info(
                f"Dropping likely-hallucinated segment "
                f"(no_speech_prob={segment.no_speech_prob:.2f}, avg_logprob={segment.avg_logprob:.2f}): "
                f"'{segment.text.strip()}'"
            )
            continue
        kept_segments.append(segment.text)
    text = "".join(kept_segments).strip()
    logger.info(
        f"Done. Detected language={info.language} "
        f"(confidence={info.language_probability:.2f}), text length={len(text)}"
    )
    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration_seconds": info.duration,
    }


class JobCanceled(Exception):
    """Raised internally to unwind out of a transcription that was canceled mid-run."""
    pass


def _run_job(job_id: str, tmp_path: str, forced_language: Optional[str]):
    """Runs on a transcribe_executor worker thread for the async job flow."""
    with jobs_lock:
        if jobs[job_id]["canceled"]:
            jobs[job_id]["status"] = "canceled"
            jobs[job_id]["finished_at"] = time.time()
            Path(tmp_path).unlink(missing_ok=True)
            return
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["started_at"] = time.time()
    try:
        result = _transcribe_file_sync(tmp_path, forced_language, job_id=job_id)
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["result"] = result
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["finished_at"] = time.time()
    except JobCanceled:
        with jobs_lock:
            jobs[job_id]["status"] = "canceled"
            jobs[job_id]["finished_at"] = time.time()
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["finished_at"] = time.time()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _save_upload_to_temp(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    suffix = Path(file.filename).suffix or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        tmp.write(content)
        return tmp.name

# ---------------------------------------------------------------------------
# Gemini — called from the SERVER, so the key lives only here, never in the
# Android app. Read from an environment variable, never hardcoded.
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash-lite"  # same model the Android app was using directly before
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"


async def call_gemini(prompt: str, json_mode: bool = False) -> str:
    """Calls Gemini's generateContent endpoint and returns the raw text of the first candidate."""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set on the server. Set it as an environment variable before starting.",
        )

    url = f"{GEMINI_BASE_URL}/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if json_mode:
        body["generationConfig"] = {"responseMimeType": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=body)

    if response.status_code != 200:
        logger.error(f"Gemini request failed: HTTP {response.status_code} - {response.text}")
        raise HTTPException(status_code=502, detail=f"Gemini request failed: HTTP {response.status_code}")

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected Gemini response shape: {data}")
        raise HTTPException(status_code=502, detail="Unexpected response from Gemini") from e


@app.get("/health")
def health():
    with jobs_lock:
        active_jobs = sum(1 for j in jobs.values() if j["status"] in ("queued", "processing"))
    return {
        "status": "ok",
        "whisper_model": MODEL_SIZE,
        "gemini_key_configured": bool(GEMINI_API_KEY),
        "max_transcribe_workers": MAX_TRANSCRIBE_WORKERS,
        "active_jobs": active_jobs,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form(default="")):
    """
    Accepts an audio file and returns its transcript — fully local, no API
    key needed. If [language] is given (a Whisper language code like "ar",
    "de", "en"), it's forced instead of letting Whisper guess — guessing
    can go wrong on noisy/quiet audio and is a common cause of both wrong-
    language output and hallucinated text.

    Still waits for the result on this same connection (unchanged
    behavior) — but the actual work now runs through the bounded
    transcribe_executor pool, so it queues behind other in-flight
    transcriptions instead of running unbounded alongside them.
    """
    tmp_path = await _save_upload_to_temp(file)
    forced_language = language.strip() or None
    logger.info(
        f"Received '{file.filename}' ({Path(tmp_path).stat().st_size} bytes), "
        f"transcribing... (forced_language={forced_language or 'auto-detect'})"
    )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(transcribe_executor, _transcribe_file_sync, tmp_path, forced_language)
        return result
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/transcribe/start", status_code=202)
async def transcribe_start(file: UploadFile = File(...), language: str = Form(default="")):
    """
    Queues a transcription job and returns immediately with a job_id —
    doesn't hold the connection open for the whole transcription like
    /transcribe does. Poll /transcribe/status/{job_id} for the result.
    Recommended for long recordings and unreliable mobile connections.
    """
    _cleanup_old_jobs()
    tmp_path = await _save_upload_to_temp(file)
    forced_language = language.strip() or None
    job_id = str(uuid.uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "result": None,
            "error": None,
            "progress": 0.0,
            "canceled": False,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
        }

    logger.info(f"Queued job {job_id} for '{file.filename}' (forced_language={forced_language or 'auto-detect'})")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(transcribe_executor, _run_job, job_id, tmp_path, forced_language)
    return {"job_id": job_id}


@app.get("/transcribe/status/{job_id}")
def transcribe_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id (it may have expired — jobs are kept for 1 hour after finishing)")
    return {
        "job_id": job_id,
        "status": job["status"],  # "queued" | "processing" | "done" | "failed" | "canceled"
        "progress": job["progress"],  # 0.0-1.0, real segment-based progress once processing starts
        "result": job["result"],
        "error": job["error"],
    }


@app.post("/transcribe/cancel/{job_id}")
def transcribe_cancel(job_id: str):
    """
    Marks a job canceled. If it hasn't started yet, the worker skips it
    entirely once it's picked up. If it's already processing, it stops at
    the next segment boundary rather than instantly — Python can't forcibly
    kill a running thread — so cancellation may take a few seconds on a
    long segment, not zero.
    """
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        if job["status"] in ("done", "failed", "canceled"):
            return {"job_id": job_id, "status": job["status"], "message": "Job already finished — nothing to cancel."}
        job["canceled"] = True
    logger.info(f"Cancel requested for job {job_id}")
    return {"job_id": job_id, "status": "canceling"}


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    target_language: str  # e.g. "German", "Arabic" — a plain language name


@app.post("/translate")
async def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Nothing to translate")

    prompt = f"""Translate the following text into {req.target_language}.
Preserve the meaning and tone. Return ONLY the translated text — no quotes,
no explanation, no original text.

Text:
{req.text}"""

    translated = await call_gemini(prompt)
    return {"translated_text": translated.strip()}


# ---------------------------------------------------------------------------
# Analysis: summary, key points, tasks, glossary, quiz — same shape the
# Android app already expects from its old direct-Gemini path.
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str
    language: str  # e.g. "ar", "de", "en" — used to keep the output in the same language


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Nothing to analyze")

    prompt = f"""Analyze the following transcript (language: {req.language}). Return ONLY valid JSON
with keys: title (string), category (string), summary (string), keyPoints (string[]), tasks (string[]),
decisions (string[]), questions (string[]), studySections (array of
{{heading, content}}), glossary (array of {{term, definition}}), quiz
(array of {{question, answer}}), topicOutline (string[]).

CRITICAL LANGUAGE RULE: every text field above MUST be written in that exact
same language ({req.language}) as the transcript, with NO exceptions and NO
translation into English or any other language.

"category" is a single short label (1-2 words): Meeting, Lecture, Task List,
Idea, Personal Note, Call, or another fitting label.

"summary" should be 3 to 6 full sentences genuinely recapping what was
discussed, why, who was involved, and any outcome.

studySections/glossary/quiz/topicOutline are ONLY for lecture/study-style
content — return empty arrays for all four otherwise.

Transcript:
{req.text}"""

    raw = await call_gemini(prompt, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Gemini did not return valid JSON: {raw}")
        raise HTTPException(status_code=502, detail="Analysis response wasn't valid JSON")


# ---------------------------------------------------------------------------
# Q&A: ask a question about a specific recording's transcript.
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    transcript: str
    question: str
    language: Optional[str] = None  # answer language; defaults to the transcript's own language


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="No question provided")

    lang_instruction = (
        f"Answer in {req.language}." if req.language
        else "Answer in the same language as the transcript."
    )
    prompt = f"""Answer the question below using ONLY information from this transcript.
If the answer isn't in the transcript, say so plainly rather than guessing. {lang_instruction}

Transcript:
{req.transcript}

Question: {req.question}"""

    answer = await call_gemini(prompt)
    return {"answer": answer.strip()}

