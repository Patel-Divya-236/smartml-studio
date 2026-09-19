# Container image for the SmartML Studio API.
#
# The backend is deliberately not serverless: it holds pipeline state between requests,
# runs training on a background thread, and serves training progress over a WebSocket.
# It wants one long-lived process with a writable volume, which is what this image plus
# the compose file provide.

FROM python:3.11-slim

# libgomp is required by LightGBM and XGBoost at runtime; the slim image omits it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so editing application code does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY src/ src/
COPY config/ config/

# Session checkpoints and logs. Both are mounted as volumes in compose — the checkpoint
# directory MUST outlive the container, or a restart loses every pipeline again, which is
# the failure this whole deployment exists to stop.
ENV SMARTML_SESSION_DIR=/data/sessions
RUN mkdir -p /data/sessions /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# --timeout-keep-alive is raised above the default 5s so the reverse proxy can hold
# connections open between a browser's steps instead of reconnecting for each one.
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "75", \
     "--timeout-graceful-shutdown", "30"]
