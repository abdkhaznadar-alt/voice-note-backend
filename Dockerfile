# ---------------------------------------------------------------------------
# Stage 1: convert the free, dialect-tuned Arabic Whisper model
# (oddadmix/whisper-small-arabic-dialectal-v2) from Hugging Face Transformers
# format into CTranslate2 format, so the running service can load it
# directly with faster-whisper — no runtime download, no per-call API cost.
#
# This stage needs torch + transformers only to run the one-time conversion;
# they are NOT copied into the final image, which keeps the deployed
# container small and fast to start.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS converter

WORKDIR /convert
COPY requirements-convert.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-convert.txt

# File names checked against the actual Hugging Face repo contents: it ships
# tokenizer_config.json / processor_config.json (not the more common
# preprocessor_config.json / special_tokens_map.json names). faster-whisper
# falls back to its own default mel-filter config when no
# preprocessor_config.json is present, which is correct for standard Whisper
# audio preprocessing, so nothing else needs to be copied.
# If the conversion fails for any reason (e.g. a future incompatible
# ctranslate2/torch build), don't fail the whole deploy: leave the output
# dir empty and let main.py fall back to the stock Whisper model at runtime.
RUN (ct2-transformers-converter \
    --model oddadmix/whisper-small-arabic-dialectal-v2 \
    --output_dir /convert/whisper-small-arabic-dialectal-v2-ct2 \
    --copy_files tokenizer_config.json \
    --quantization int8 \
    --force) || (echo "Dialect model conversion failed — will fall back to stock Whisper at runtime" && mkdir -p /convert/whisper-small-arabic-dialectal-v2-ct2)

# ---------------------------------------------------------------------------
# Stage 2: the actual runtime image (small — no torch/transformers here)
# ---------------------------------------------------------------------------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=converter /convert/whisper-small-arabic-dialectal-v2-ct2 /app/models/whisper-small-arabic-dialectal-v2-ct2

COPY main.py .

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache/huggingface

ENV PORT=8000
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
