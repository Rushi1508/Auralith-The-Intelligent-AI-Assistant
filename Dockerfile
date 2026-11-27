FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/hf TIKTOKEN_CACHE_DIR=/hf/tiktoken FASTEMBED_CACHE_PATH=/hf/fastembed
COPY requirements.txt .
# CPU-only torch first; PyPI's Linux wheel drags in several GB of CUDA libs
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.0 \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
# Workers are stateless (registry + vectors in Postgres); each loads its own models,
# so budget memory for 2-4 workers. uvicorn reads WEB_CONCURRENCY.
ENV WEB_CONCURRENCY=2
CMD ["uvicorn", "auralith.api:app", "--host", "0.0.0.0", "--port", "8000"]
