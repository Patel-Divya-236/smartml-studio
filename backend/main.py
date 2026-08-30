"""FastAPI application for SmartML Studio.

Every route imports its logic from `src/`, which is unchanged by the migration. The
backend is a transport layer over the existing profiler, advisors, pipelines, trainer,
ensemble and explainer — the pipeline's behaviour is defined there, not here.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import advisors, datasets, evaluation, pipeline, predictions, training
from backend.core.secrets import load_llm_env
from backend.core.session import STORE
from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load LLM credentials from the secrets file, if one is present."""
    load_llm_env()
    yield


app = FastAPI(
    title="SmartML Studio API",
    version="2.0.0",
    description="Backend for the React ML workspace.",
    lifespan=lifespan,
)

# The Vite dev server runs on a different port, so the browser treats API calls as
# cross-origin. Origins are listed explicitly rather than "*" because credentials and the
# session header are involved.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
extra = os.getenv("SMARTML_ALLOWED_ORIGINS")
if extra:
    ALLOWED_ORIGINS.extend(o.strip() for o in extra.split(",") if o.strip())

# Vercel gives one deployment several hostnames -- the project alias, an account-scoped
# alias, and a unique URL per deployment -- and mints a fresh one for every preview build.
# Pinning them by hand means the API breaks quietly on the next deploy, so the project's
# own subdomains are matched by pattern instead. The pattern is anchored to this project's
# name: it is deliberately not a blanket `*.vercel.app`, which would let any site hosted
# on Vercel call this API.
VERCEL_ORIGIN_PATTERN = os.getenv(
    "SMARTML_ALLOWED_ORIGIN_REGEX",
    r"^https://smartml-studio[a-z0-9-]*\.vercel\.app$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=VERCEL_ORIGIN_PATTERN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    datasets.router,
    advisors.router,
    pipeline.router,
    training.router,
    evaluation.router,
    predictions.router,
):
    app.include_router(router)


@app.post("/api/session")
def create_session() -> dict:
    """Issue a session id for the browser to send as `X-Session-Id`."""
    STORE.purge_expired()
    session = STORE.create()
    return {"session_id": session.id, "completed_steps": session.completed_steps()}


@app.get("/api/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}
