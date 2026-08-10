from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from io import BytesIO
import wave

import numpy as np

from .config import Settings
from .contract import SAMPLE_RATE, SAMPLES_PER_CHUNK


class MediaValidationError(ValueError):
    pass


@dataclass(frozen=True)
class MultipartChunk:
    sequence: int
    frame: bytes
    audio: bytes


@dataclass(frozen=True)
class DecodedChunk:
    image: object
    waveform: np.ndarray


def parse_multipart_chunk(content_type: str, body: bytes, settings: Settings) -> MultipartChunk:
    """Parse the three accepted multipart fields entirely in memory."""

    if len(body) > settings.max_request_bytes:
        raise MediaValidationError("multipart body exceeds the configured limit")
    if not content_type.lower().startswith("multipart/form-data;"):
        raise MediaValidationError("Content-Type must be multipart/form-data")
    envelope = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
        + body
    )
    message = BytesParser(policy=policy.default).parsebytes(envelope)
    if not message.is_multipart():
        raise MediaValidationError("malformed multipart body")
    fields: dict[str, tuple[str, bytes]] = {}
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            raise MediaValidationError("all multipart parts must be form-data")
        name = part.get_param("name", header="content-disposition")
        if name not in {"sequence", "frame", "audio"} or name in fields:
            raise MediaValidationError("multipart fields must be exactly sequence, frame, audio")
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            raise MediaValidationError(f"could not decode multipart field {name}")
        fields[name] = (part.get_content_type().lower(), payload)
    if set(fields) != {"sequence", "frame", "audio"}:
        raise MediaValidationError("multipart fields must be exactly sequence, frame, audio")
    try:
        sequence_text = fields["sequence"][1].decode("ascii", errors="strict")
        if not sequence_text.isdecimal() or len(sequence_text) > 2:
            raise ValueError
        sequence = int(sequence_text)
    except ValueError as exc:
        raise MediaValidationError("sequence must be a nonnegative decimal integer") from exc
    frame_type, frame = fields["frame"]
    audio_type, audio = fields["audio"]
    if frame_type not in {"image/jpeg", "image/jpg"}:
        raise MediaValidationError("frame must have Content-Type image/jpeg")
    if audio_type not in {"audio/wav", "audio/x-wav"}:
        raise MediaValidationError("audio must have Content-Type audio/wav")
    if not 0 < len(frame) <= settings.max_frame_bytes:
        raise MediaValidationError("JPEG frame size is outside the configured limit")
    if not 0 < len(audio) <= settings.max_audio_bytes:
        raise MediaValidationError("WAV chunk size is outside the configured limit")
    return MultipartChunk(sequence=sequence, frame=frame, audio=audio)


def decode_chunk(chunk: MultipartChunk, settings: Settings) -> DecodedChunk:
    """Validate/decode media from bytes; neither input is written to disk."""

    try:
        from PIL import Image

        with Image.open(BytesIO(chunk.frame)) as opened:
            if opened.format != "JPEG":
                raise MediaValidationError("frame is not a JPEG bitstream")
            width, height = opened.size
            if width < 16 or height < 16 or width * height > settings.max_frame_pixels:
                raise MediaValidationError("JPEG dimensions are outside the configured limit")
            opened.verify()
        with Image.open(BytesIO(chunk.frame)) as reopened:
            image = reopened.convert("RGB").copy()
    except MediaValidationError:
        raise
    except Exception as exc:
        raise MediaValidationError("invalid JPEG frame") from exc

    try:
        with wave.open(BytesIO(chunk.audio), "rb") as wav:
            if wav.getnchannels() != 1:
                raise MediaValidationError("WAV must be mono")
            if wav.getsampwidth() != 2:
                raise MediaValidationError("WAV must use 16-bit PCM")
            if wav.getframerate() != SAMPLE_RATE:
                raise MediaValidationError("WAV must be sampled at 16 kHz")
            if wav.getcomptype() != "NONE":
                raise MediaValidationError("WAV must be uncompressed PCM")
            if wav.getnframes() != SAMPLES_PER_CHUNK:
                raise MediaValidationError("WAV must contain exactly one second (16000 samples)")
            pcm = wav.readframes(SAMPLES_PER_CHUNK)
            if wav.readframes(1):
                raise MediaValidationError("WAV contains trailing audio samples")
    except MediaValidationError:
        raise
    except (EOFError, wave.Error) as exc:
        raise MediaValidationError("invalid WAV chunk") from exc
    waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if waveform.shape != (SAMPLES_PER_CHUNK,) or not np.isfinite(waveform).all():
        raise MediaValidationError("invalid PCM sample payload")
    return DecodedChunk(image=image, waveform=waveform)
