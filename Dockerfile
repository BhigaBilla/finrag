# FinRAG API on CPU. The index is not baked in: mount data/ (pages.jsonl,
# chunks.jsonl, index/) at /app/data and pass the keys with --env-file .env.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build (~2 GB smaller)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY eval/questions.jsonl eval/
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
