# DHAWQ API. Deliberately WITHOUT the pipeline stack — no torch, no open_clip,
# no umap. pipelines/ is build-time only and import-forbidden from the service,
# so the server image stays small and the §4 boundary is enforced by what is
# not installed as well as by CI.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ ./services/
COPY eval/ ./eval/
COPY pipelines/manifests/ ./pipelines/manifests/
COPY data/processed/ ./data/processed/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn services.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
