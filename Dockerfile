FROM python:3.11-slim

# ffmpeg is required by faster-whisper to decode audio (m4a, webm, mp3, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render assigns a PORT dynamically via environment variable (not a fixed
# port like Hugging Face's 7860) — default to 8000 for local `docker run`
# testing, but Render overrides this at runtime.
ENV PORT=8000
EXPOSE 8000

# A writable cache dir for the downloaded Whisper model weights — the
# container may run as a non-root user, so /root/.cache isn't writable;
# point the cache somewhere inside /app instead.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Shell form (not exec-array form) so $PORT actually gets substituted at
# container start — Render sets this to whatever port it wants to route to.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
