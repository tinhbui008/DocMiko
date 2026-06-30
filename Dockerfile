# ============================================================
# Mikotech PDF Translator - CLI batch image (Ubuntu/Debian base)
# Bakes OCR models into the image so the server can run offline.
# ============================================================
FROM python:3.11-slim

# --- System libraries required at runtime ---
#  libgomp1            -> paddlepaddle / onnxruntime (OpenMP)
#  libglib2.0-0,libgl1 -> opencv (even the headless build links libgl on some wheels)
#  tesseract-ocr*      -> pytesseract fallback path
#  fonts-* + others    -> safety fallbacks; project ships its own fonts in ./fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# --- Python deps first (better layer caching) ---
# build-essential is needed to compile C extensions (e.g. stringzilla, a
# transitive dep of paddleocr); it is purged afterwards to keep the image lean.
COPY requirements-docker.txt /app/requirements-docker.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip \
    && pip install -r /app/requirements-docker.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Bake OCR models into the image (build needs internet, runtime does not) ---
COPY warmup_ocr.py /app/warmup_ocr.py
RUN python /app/warmup_ocr.py || echo "WARN: OCR warmup skipped (models will download at runtime if missing)"

# --- Application source ---
COPY . /app

# Normalize shell script line endings (in case they were committed with CRLF)
RUN for s in /app/*.sh; do sed -i 's/\r$//' "$s"; chmod +x "$s"; done

# Input/output mount points
RUN mkdir -p /app/input /app/output

ENTRYPOINT ["/bin/bash", "run_pipeline.sh"]
