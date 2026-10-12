FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/hf HF_HUB_DISABLE_SYMLINKS_WARNING=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# The pip CUDA libraries are not on the loader path by default
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
COPY app.py .
RUN mkdir -p /hf && chmod 777 /hf
EXPOSE 9000
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s \
    CMD curl -fsS http://127.0.0.1:9000/healthz || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
