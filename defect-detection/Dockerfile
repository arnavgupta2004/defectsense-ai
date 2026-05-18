FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY training ./training
COPY scripts ./scripts
COPY data/download_mvtec.py ./data/download_mvtec.py

RUN mkdir -p /app/data/uploads /app/data/custom/train/good /app/artifacts

# Low-memory profile: bake model at build time so Render startup stays under 512 MB.
ENV LOW_MEMORY=1 \
    RENDER=true \
    IMAGE_SIZE=128 \
    CORESET_SAMPLING_RATIO=0.04 \
    INFERENCE_CHUNK_SIZE=256 \
    TRAIN_BATCH_SIZE=2 \
    BOOTSTRAP_TRAIN_IMAGES=6 \
    FEATURE_LAYERS=layer3 \
    NUM_NEIGHBORS=3 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN python scripts/bootstrap_data.py && python -m training.train_patchcore

COPY .env.example .env
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

ENV APP_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    DB_URL=sqlite:////app/data/defects.db \
    UPLOAD_DIR=/app/data/uploads \
    TRAIN_DIR=/app/data/custom/train/good \
    MODEL_MEMORY_BANK_PATH=/app/artifacts/patchcore_memory_bank.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
