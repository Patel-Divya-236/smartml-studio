"""FastAPI application for SmartML Studio.

Every route imports its logic from `src/`, which is unchanged by the migration. The
backend is a transport layer over the existing profiler, advisors, pipelines, trainer,
ensemble and explainer — the pipeline's behaviour is defined there, not here.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import advisors, datasets, evaluation, pipeline, predictions, training
from backend.core.secrets import load_llm_env
from backend.core.session import STORE
from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Sessions hold DataFrames and fitted models, and now leave files on disk too. Sweeping
# only when a new session is created meant an idle server kept every one of them; an hour
# is frequent enough to bound that without doing noticeable work.
PURGE_INTERVAL_SECONDS = 60 * 60


async def _purge_loop() -> None:
    """Drop expired sessions and their checkpoints on a timer, for as long as we run."""
    while True:
        await asyncio.sleep(PURGE_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(STORE.purge_expired)
        except Exception:
            # A failed sweep must not kill the loop; the next tick tries again.
            logger.exception("Session purge failed")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load LLM credentials, then run the session sweeper for the process's lifetime."""
    load_llm_env()
    STORE.purge_expired()  # clear checkpoints orphaned by however the last process ended
    sweeper = asyncio.create_task(_purge_loop())
    try:
        yield
    finally:
        sweeper.cancel()
        with suppress(asyncio.CancelledError):
            await sweeper


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
