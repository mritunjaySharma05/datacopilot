# DataCopilot images: one Dockerfile, two targets.
#   docker build -f docker/Dockerfile --target api -t datacopilot-api .
#   docker build -f docker/Dockerfile --target ui  -t datacopilot-ui  .
# For the cloud (x86 servers) from an ARM Mac add: --platform linux/amd64
#
# Images contain code and the 3 KB answerability model only: no data (mounted at /data
# or synced from S3), no .env and no API keys (passed as environment variables at run time).

ARG PYTHON_IMAGE=python:3.13.18-slim-bookworm

# ---------- shared base: non-root user, no .pyc files, unbuffered logs ----------
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src
RUN useradd --create-home --uid 10001 app
WORKDIR /app

# ---------- API: FastAPI + pipeline ----------
FROM base AS api
# libgomp1: OpenMP runtime that the xgboost wheel needs on Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-api.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app models/answerability.joblib /app/models/answerability.joblib
ENV DATA_DIR=/data \
    LLM_PROVIDER=ollama
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]
# --no-access-log: every /ask writes its own JSON line (request_log.py); health checks stay quiet
CMD ["uvicorn", "datacopilot.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

# ---------- UI: Streamlit, talks to the API over HTTP ----------
FROM base AS ui
COPY docker/requirements-ui.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# The UI needs only its own package (no pipeline code in this image).
COPY --chown=app:app src/datacopilot/__init__.py /app/src/datacopilot/__init__.py
COPY --chown=app:app src/datacopilot/ui/ /app/src/datacopilot/ui/
ENV API_URL=http://api:8000
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"]
CMD ["streamlit", "run", "/app/src/datacopilot/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
