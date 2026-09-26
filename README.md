# VoiceNote AI Backend

FastAPI + faster-whisper (Arabic-dialect fine-tuned model) + Gemini (server-side only).

## What changed: free Arabic-dialect transcription

The backend now uses **`oddadmix/whisper-small-arabic-dialectal-v2`**, a free,
open-source Whisper model fine-tuned specifically on Arabic dialects (Gulf,
Levantine, Egyptian, Iraqi, etc.), instead of the stock "small" model. This
directly targets the "يفرغ ومفهوم وبلا تكلفة" (accurate/understandable + no
cost) requirement — it's still local Whisper inference, so there is no extra
per-request cost, just noticeably better dialect accuracy than stock Whisper.

Because this model ships in Hugging Face Transformers format, the Dockerfile
converts it to CTranslate2 format (the format `faster-whisper` needs) at
**build time**, using a multi-stage build:

- **Stage 1 (`converter`)**: installs `transformers` + `torch` (CPU) +
  `ctranslate2`, downloads the model from Hugging Face, and runs
  `ct2-transformers-converter` to produce a local CTranslate2 model
  directory.
- **Stage 2 (final image)**: only installs the lightweight runtime deps
  (`fastapi`, `faster-whisper`, etc. — no `torch`/`transformers`), and copies
  the already-converted model directory from Stage 1.

This keeps the deployed container small and fast to start, while the
one-time conversion happens automatically on every image build — no manual
steps needed. `main.py` loads the converted model automatically if it finds
it at `/app/models/whisper-small-arabic-dialectal-v2-ct2`; otherwise it logs
a warning and falls back to the stock `"small"` model, so the service never
fails to start even if the conversion step is skipped for some reason.

### Config

- `USE_DIALECT_MODEL` (env var, default `true`) — set to `false` to force
  the stock Whisper model instead.
- `GEMINI_API_KEY` (env var, required for `/translate`, `/analyze`, `/ask`,
  and combined `also_analyze` jobs) — **never commit this**, set it only in
  your hosting platform's environment variables (Render/Railway dashboard).

### Deploy notes

- The build is heavier than before (it briefly installs `torch` CPU wheels
  in the converter stage), so the **build** may take a few minutes longer
  and use more temporary disk than the previous version. The **runtime**
  image stays lightweight since torch/transformers are not carried into the
  final stage.
- If your platform's build step times out or runs out of disk on the free
  tier, the two ways out are: (1) upgrade the build machine tier temporarily
  just for this one deploy, or (2) set `USE_DIALECT_MODEL=false` and
  redeploy without the conversion stage (falls back to stock `"small"`,
  same as before).

## Endpoints

- `GET /health`
- `POST /transcribe` (legacy, synchronous)
- `POST /transcribe/start` → `{job_id}` (async job, form fields: `file`,
  `language`, `also_analyze`)
- `GET /transcribe/status/{job_id}`
- `POST /transcribe/cancel/{job_id}`
- `POST /translate`
- `POST /analyze`
- `POST /ask`
