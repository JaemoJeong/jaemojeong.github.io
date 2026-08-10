from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import secrets
import time
from typing import Any, Callable


class SessionError(RuntimeError):
    status_code = 400


class SessionNotFound(SessionError):
    status_code = 404


class SequenceConflict(SessionError):
    status_code = 409


class SessionCapacity(SessionError):
    status_code = 429


@dataclass
class StreamSession:
    session_id: str
    engine_state: Any
    created_at: float
    last_seen: float
    next_sequence: int = 0
    accepted_chunks: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    def __init__(
        self,
        *,
        state_factory: Callable[[], Any],
        ttl_seconds: int,
        max_sessions: int,
        max_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state_factory = state_factory
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._max_seconds = max_seconds
        self._clock = clock
        self._sessions: dict[str, StreamSession] = {}

    def _purge(self) -> None:
        now = self._clock()
        expired = [
            key
            for key, session in self._sessions.items()
            if not session.lock.locked()
            and (
                now - session.created_at > self._ttl
                or now - session.last_seen > self._ttl
            )
        ]
        for key in expired:
            del self._sessions[key]

    def create(self) -> StreamSession:
        self._purge()
        if len(self._sessions) >= self._max_sessions:
            raise SessionCapacity("too many active streaming sessions")
        now = self._clock()
        session_id = secrets.token_urlsafe(24)
        session = StreamSession(session_id, self._state_factory(), now, now)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> StreamSession:
        self._purge()
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFound("streaming session was not found or expired")
        return session

    def reset(self, session_id: str) -> StreamSession:
        session = self.get(session_id)
        session.engine_state = self._state_factory()
        session.next_sequence = 0
        session.last_seen = self._clock()
        return session

    def delete(self, session_id: str) -> None:
        self._purge()
        if self._sessions.pop(session_id, None) is None:
            raise SessionNotFound("streaming session was not found or expired")

    def validate_sequence(self, session: StreamSession, sequence: int) -> None:
        if sequence != session.next_sequence:
            raise SequenceConflict(
                f"expected sequence {session.next_sequence}, received {sequence}"
            )
        if sequence >= self._max_seconds or session.accepted_chunks >= self._max_seconds:
            raise SequenceConflict(f"session is limited to {self._max_seconds} one-second chunks")

    def commit(self, session: StreamSession, next_state: Any) -> None:
        session.engine_state = next_state
        session.next_sequence += 1
        session.accepted_chunks += 1
        session.last_seen = self._clock()

    @property
    def active_count(self) -> int:
        self._purge()
        return len(self._sessions)
