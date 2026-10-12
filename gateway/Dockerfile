FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY app/requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
EXPOSE 11434
HEALTHCHECK --interval=15s --timeout=4s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/healthz', timeout=3)"
CMD ["uvicorn", "--app-dir", "app", "main:app", "--host", "0.0.0.0", "--port", "11434", "--no-access-log"]
