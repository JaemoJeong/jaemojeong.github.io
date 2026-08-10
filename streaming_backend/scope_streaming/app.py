from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .config import Settings
from .contract import LLP_CLASSES, SCHEMA_VERSION
from .engines import MockEngine, build_engine
from .media import MediaValidationError, decode_chunk, parse_multipart_chunk
from .sessions import SequenceConflict, SessionError, SessionStore


async def _bounded_body(request: Request, limit: int) -> bytes:
    header = request.headers.get("content-length")
    if header is not None:
        try:
            declared = int(header)
        except ValueError as exc:
            raise MediaValidationError("invalid Content-Length") from exc
        if declared < 0 or declared > limit:
            raise MediaValidationError("request body exceeds the configured limit")
    pieces: list[bytes] = []
    total = 0
    async for piece in request.stream():
        total += len(piece)
        if total > limit:
            raise MediaValidationError("request body exceeds the configured limit")
        pieces.append(piece)
    return b"".join(pieces)


def create_app(
    settings: Settings | None = None,
    *,
    engine: MockEngine | Any | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime = engine if engine is not None else build_engine(configured)
        application.state.engine = runtime
        application.state.engine_lock = asyncio.Lock()
        application.state.sessions = SessionStore(
            state_factory=runtime.new_state,
            ttl_seconds=configured.session_ttl_seconds,
            max_sessions=configured.max_sessions,
            max_seconds=configured.max_seconds,
        )
        yield

    application = FastAPI(
        title="SCoPE exploratory streaming API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @application.exception_handler(SessionError)
    async def session_error_handler(_: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "schema_version": SCHEMA_VERSION,
                "error": str(exc),
                "detail": str(exc),
            },
        )

    @application.exception_handler(MediaValidationError)
    async def media_error_handler(_: Request, exc: MediaValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "schema_version": SCHEMA_VERSION,
                "error": str(exc),
                "detail": str(exc),
            },
        )

    @application.get("/v1/health")
    async def health(request: Request) -> dict[str, Any]:
        runtime = request.app.state.engine
        sessions: SessionStore = request.app.state.sessions
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "mode": runtime.mode,
            "model": runtime.model_name,
            "active_sessions": sessions.active_count,
        }

    @application.post("/v1/sessions", status_code=201)
    async def start_session(request: Request) -> dict[str, Any]:
        body = await _bounded_body(request, 256)
        if body.strip() not in {b"", b"{}"}:
            return JSONResponse(
                status_code=422,
                content={
                    "schema_version": SCHEMA_VERSION,
                    "error": "session start body must be an empty JSON object",
                    "detail": "session start body must be an empty JSON object",
                },
            )
        sessions: SessionStore = request.app.state.sessions
        session = sessions.create()
        runtime = request.app.state.engine
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": session.session_id,
            "status": "running",
            "mode": runtime.mode,
            "model": runtime.model_name,
            "next_sequence": 0,
            "max_seconds": configured.max_seconds,
            "classes": list(LLP_CLASSES),
        }

    @application.post("/v1/sessions/{session_id}/chunks")
    async def predict_chunk(session_id: str, request: Request) -> dict[str, Any]:
        total_start = time.perf_counter()
        sessions: SessionStore = request.app.state.sessions
        session = sessions.get(session_id)
        if session.lock.locked():
            raise SequenceConflict("another chunk is already in flight for this session")
        async with session.lock:
            body = await _bounded_body(request, configured.max_request_bytes)
            multipart = parse_multipart_chunk(
                request.headers.get("content-type", ""), body, configured
            )
            sessions.validate_sequence(session, multipart.sequence)
            decode_start = time.perf_counter()
            media = decode_chunk(multipart, configured)
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            runtime = request.app.state.engine
            async with request.app.state.engine_lock:
                output, next_state, timings = await asyncio.to_thread(
                    runtime.predict, session.engine_state, media
                )
            sessions.commit(session, next_state)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "sequence": multipart.sequence,
            "time_start_seconds": multipart.sequence,
            "time_end_seconds": multipart.sequence + 1,
            "classes": list(LLP_CLASSES),
            "timing_ms": {
                "decode": float(decode_ms),
                "encode": float(timings["encode"]),
                "inference": float(timings["inference"]),
                "total": float(total_ms),
            },
            **output,
        }

    @application.post("/v1/sessions/{session_id}/reset")
    async def reset_session(session_id: str, request: Request) -> dict[str, Any]:
        sessions: SessionStore = request.app.state.sessions
        session = sessions.get(session_id)
        async with session.lock:
            reset = sessions.reset(session_id)
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": reset.session_id,
            "status": "running",
            "next_sequence": 0,
        }

    @application.delete("/v1/sessions/{session_id}", status_code=204)
    async def stop_session(session_id: str, request: Request) -> Response:
        sessions: SessionStore = request.app.state.sessions
        session = sessions.get(session_id)
        async with session.lock:
            sessions.delete(session_id)
        return Response(status_code=204)

    return application


app = create_app()
