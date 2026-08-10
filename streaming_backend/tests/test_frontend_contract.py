from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import wave

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scope_streaming.app import create_app
from scope_streaming.config import Settings
from scope_streaming.engines import MockEngine


DEMO_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = DEMO_ROOT / "index.html"
LLP_CLASSES = [
    "Speech", "Car", "Cheering", "Dog", "Cat", "Frying_(food)",
    "Basketball_bounce", "Fire_alarm", "Chainsaw", "Cello", "Banjo",
    "Singing", "Chicken_rooster", "Violin_fiddle", "Vacuum_cleaner",
    "Baby_laughter", "Accordion", "Lawn_mower", "Motorcycle", "Helicopter",
    "Acoustic_guitar", "Telephone_bell_ringing", "Baby_cry_infant_cry",
    "Blender", "Clapping",
]


class ProductionContractFixtureEngine(MockEngine):
    mode = "languagebind"
    model_name = "LanguageBind_unit (Audio_FT + Image)"

    def predict(self, state, media):
        output, next_state, timings = super().predict(state, media)
        output["dense"]["score_kind"] = "class_axis_zscore_sigmoid"
        output["scope"]["score_kind"] = "raw_nonnegative_stage2_weight"
        output["scope"]["prefix_segments"] = next_state.length
        return output, next_state, timings


def backend_settings() -> Settings:
    return Settings(
        mode="mock",
        scope_repo=BACKEND_ROOT,
        languagebind_code=None,
        languagebind_vendor=None,
        languagebind_cache=None,
        audio_prototypes=None,
        visual_prototypes=None,
        audio_mean=None,
        visual_mean=None,
        device="cpu",
        cors_origins=("https://example.test",),
        session_ttl_seconds=30,
        max_sessions=2,
    )


def jpeg_bytes() -> bytes:
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (64, 48), (30, 120, 220)).save(output, format="JPEG")
    return output.getvalue()


def wav_bytes() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000)
    return output.getvalue()


def branch() -> dict[str, list[float] | list[bool]]:
    return {"scores": [0.0] * 25, "predictions": [False] * 25}


def payload() -> dict[str, object]:
    branches = {name: branch() for name in ("audio", "visual", "audio_visual")}
    return {
        "schema_version": "scope-streaming.v1",
        "sequence": 0,
        "time_start_seconds": 0,
        "time_end_seconds": 1,
        "classes": LLP_CLASSES,
        "dense": {
            "threshold": 0.85,
            "comparator": ">",
            "score_kind": "class_axis_zscore_sigmoid",
            "branches": branches,
        },
        "scope": {
            "causal": True,
            "future_segments_used": 0,
            "prefix_segments": 1,
            "score_kind": "raw_nonnegative_stage2_weight",
            "branches": branches,
        },
    }


class FrontendValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("node") is None:
            raise unittest.SkipTest("Node.js is required to execute the page validator")
        html = INDEX_HTML.read_text(encoding="utf-8")
        constants_start = html.index('const LIVE_SCHEMA_VERSION = "scope-streaming.v1";')
        constants_end = html.index("const LIVE_EMPTY_COPY", constants_start)
        functions_start = html.index("function exactLiveClasses", constants_end)
        functions_end = html.index("async function drainLiveQueue", functions_start)
        cls.validator_source = (
            html[constants_start:constants_end]
            + html[functions_start:functions_end]
        )

    def run_validator(
        self, candidate: dict[str, object], expression: str = "validateLivePayload(candidate, 0)"
    ) -> subprocess.CompletedProcess[str]:
        script = (
            self.validator_source
            + "\nconst candidate = " + json.dumps(candidate, separators=(",", ":")) + ";\n"
            + f"try {{ {expression}; process.stdout.write('accepted'); }} "
            + "catch (error) { process.stderr.write(String(error.message || error)); process.exit(2); }\n"
        )
        return subprocess.run(
            ["node", "-e", script],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_actual_page_validator_accepts_production_fixture(self) -> None:
        result = self.run_validator(payload())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "accepted")

    def test_actual_backend_response_passes_actual_page_validator(self) -> None:
        application = create_app(
            backend_settings(), engine=ProductionContractFixtureEngine()
        )
        with TestClient(application) as client:
            session = client.post("/v1/sessions", json={}).json()
            response = client.post(
                f'/v1/sessions/{session["session_id"]}/chunks',
                files={
                    "sequence": (None, "0", "text/plain"),
                    "frame": ("frame.jpg", jpeg_bytes(), "image/jpeg"),
                    "audio": ("audio.wav", wav_bytes(), "audio/wav"),
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        result = self.run_validator(response.json())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_backend_session_passes_actual_page_validator(self) -> None:
        application = create_app(
            backend_settings(), engine=ProductionContractFixtureEngine()
        )
        with TestClient(application) as client:
            session = client.post("/v1/sessions", json={})
        self.assertEqual(session.status_code, 201, session.text)
        result = self.run_validator(session.json(), "validateLiveSession(candidate)")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_page_validator_rejects_any_ground_truth_field(self) -> None:
        candidate = payload()
        candidate["ground_truth"] = None
        result = self.run_validator(candidate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incompatible response contract", result.stderr)

    def test_actual_page_validator_rejects_future_or_wrong_prefix(self) -> None:
        candidate = payload()
        candidate["scope"]["future_segments_used"] = 1  # type: ignore[index]
        result = self.run_validator(candidate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-causal", result.stderr)


if __name__ == "__main__":
    unittest.main()
