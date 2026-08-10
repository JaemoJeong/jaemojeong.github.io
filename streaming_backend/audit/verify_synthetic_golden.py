#!/usr/bin/env python3
"""Recompute and verify the rights-free production GPU sentinel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


BACKEND_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = Path(__file__).with_name(
    "synthetic_constant_rgb_zero_audio_prediction.json"
)
EXPECTED_SHA256 = (
    "95abcf9024c4b7012856d021769c8e1a8ff77ae50f980dd40b26dc7ea0e90fb6"
)
IMAGE_SHAPE = (48, 64, 3)
IMAGE_RGB = (30, 120, 220)
AUDIO_SAMPLES = 16_000


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    sys.path.insert(0, str(BACKEND_ROOT))
    from scope_streaming.config import Settings
    from scope_streaming.contract import LLP_CLASSES
    from scope_streaming.engines import LanguageBindScopeEngine
    from scope_streaming.media import DecodedChunk

    image_array = np.empty(IMAGE_SHAPE, dtype=np.uint8)
    image_array[...] = np.asarray(IMAGE_RGB, dtype=np.uint8)
    waveform = np.zeros((AUDIO_SAMPLES,), dtype=np.float32)
    image_bytes = np.ascontiguousarray(image_array).tobytes(order="C")
    audio_bytes = waveform.astype("<f4", copy=False).tobytes(order="C")

    engine = LanguageBindScopeEngine(Settings.from_env())
    output, next_state, timings = engine.predict(
        engine.new_state(),
        DecodedChunk(
            image=Image.fromarray(image_array, mode="RGB"), waveform=waveform
        ),
    )
    if next_state.length != 1:
        raise RuntimeError(f"expected one prefix row, observed {next_state.length}")

    score_hashes: dict[str, dict[str, dict[str, str]]] = {}
    selected: dict[str, dict[str, list[str]]] = {}
    for method in ("dense", "scope"):
        score_hashes[method] = {}
        selected[method] = {}
        for branch, values in output[method]["branches"].items():
            scores = np.asarray(values["scores"], dtype="<f4")
            predictions = np.asarray(values["predictions"], dtype=np.uint8)
            score_hashes[method][branch] = {
                "scores_float32_le_sha256": bytes_sha256(
                    scores.tobytes(order="C")
                ),
                "predictions_uint8_sha256": bytes_sha256(
                    predictions.tobytes(order="C")
                ),
            }
            selected[method][branch] = [
                label
                for label, active in zip(LLP_CLASSES, predictions)
                if active
            ]

    candidate = {
        "schema": "scope-streaming-synthetic-golden-prediction-v1",
        "input": {
            "image": {
                "construction": (
                    "np.empty((48,64,3),uint8); image[...] = (30,120,220)"
                ),
                "shape_hwc": list(IMAGE_SHAPE),
                "dtype": "uint8",
                "rgb": list(IMAGE_RGB),
                "byte_order": "C; HWC RGB",
                "sha256": bytes_sha256(image_bytes),
            },
            "audio": {
                "construction": "np.zeros((16000,),dtype=float32)",
                "shape": [AUDIO_SAMPLES],
                "sample_rate_hz": 16_000,
                "channels": 1,
                "dtype": "float32-le",
                "byte_order": "C",
                "sha256": bytes_sha256(audio_bytes),
            },
        },
        "output": output,
        "score_hashes": score_hashes,
        "selected_labels": selected,
    }
    candidate_bytes = canonical_bytes(candidate)
    candidate_sha = bytes_sha256(candidate_bytes)
    frozen_bytes = GOLDEN_PATH.read_bytes()
    frozen_sha = bytes_sha256(frozen_bytes)
    if frozen_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"tracked golden hash mismatch: expected {EXPECTED_SHA256}, got {frozen_sha}"
        )
    if candidate_bytes != frozen_bytes:
        raise RuntimeError(
            f"GPU golden mismatch: expected {frozen_sha}, got {candidate_sha}"
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "prediction_sha256": candidate_sha,
                "encode_ms": float(timings["encode"]),
                "inference_ms": float(timings["inference"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
