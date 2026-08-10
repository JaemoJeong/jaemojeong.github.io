from __future__ import annotations

import asyncio
from io import BytesIO
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scope_streaming.app import create_app
from scope_streaming.config import (
    FROZEN_ASSET_SHA256,
    FROZEN_CHECKPOINT_SHA256,
    Settings,
    file_sha256,
    tree_manifest_sha256,
)
from scope_streaming.contract import LLP_CLASSES, SCHEMA_VERSION
from scope_streaming.engines import (
    EngineState,
    FROZEN_RUNTIME_VERSIONS,
    MockEngine,
    SYNTHETIC_GOLDEN_SHA256,
    _load_method_modules,
    _load_prototypes,
    _validate_scope_source,
)
from scope_streaming.sessions import SequenceConflict, SessionNotFound, SessionStore


def settings() -> Settings:
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
        max_sessions=4,
    )


def jpeg_bytes() -> bytes:
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (64, 48), (30, 120, 220)).save(output, format="JPEG", quality=75)
    return output.getvalue()


def wav_bytes(*, frames: int = 16_000, rate: int = 16_000, channels: int = 1) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * frames * channels)
    return output.getvalue()


class ProductionContractFixtureEngine(MockEngine):
    """Cheap production-shaped fixture for the static frontend contract."""

    mode = "languagebind"
    model_name = "LanguageBind_unit (Audio_FT + Image)"

    def predict(self, state, media):
        output, next_state, timings = super().predict(state, media)
        output["dense"]["score_kind"] = "class_axis_zscore_sigmoid"
        output["scope"]["score_kind"] = "raw_nonnegative_stage2_weight"
        output["scope"]["prefix_segments"] = next_state.length
        return output, next_state, timings


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(create_app(settings(), engine=MockEngine()))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def start(self) -> str:
        response = self.client.post("/v1/sessions", json={})
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["classes"], list(LLP_CLASSES))
        self.assertNotIn("ground_truth", payload)
        self.assertEqual(payload["mode"], "mock")
        return payload["session_id"]

    def send(self, session_id: str, sequence: int, audio: bytes | None = None):
        return self.client.post(
            f"/v1/sessions/{session_id}/chunks",
            files={
                "sequence": (None, str(sequence), "text/plain"),
                "frame": ("frame.jpg", jpeg_bytes(), "image/jpeg"),
                "audio": ("audio.wav", audio or wav_bytes(), "audio/wav"),
            },
        )

    def test_exact_branch_and_class_contract(self) -> None:
        session_id = self.start()
        response = self.send(session_id, 0)
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["sequence"], 0)
        self.assertNotIn("ground_truth", payload)
        self.assertTrue(payload["scope"]["causal"])
        self.assertEqual(payload["scope"]["future_segments_used"], 0)
        self.assertEqual(payload["dense"]["threshold"], 0.85)
        self.assertEqual(payload["dense"]["comparator"], ">")
        for method in ("dense", "scope"):
            self.assertEqual(
                set(payload[method]["branches"]), {"audio", "visual", "audio_visual"}
            )
            for branch in payload[method]["branches"].values():
                self.assertEqual(len(branch["scores"]), 25)
                self.assertEqual(len(branch["predictions"]), 25)
        self.assertGreaterEqual(payload["timing_ms"]["total"], 0.0)

    def test_sequence_is_strict_and_reset_is_explicit(self) -> None:
        session_id = self.start()
        self.assertEqual(self.send(session_id, 0).status_code, 200)
        conflict = self.send(session_id, 0)
        self.assertEqual(conflict.status_code, 409)
        reset = self.client.post(f"/v1/sessions/{session_id}/reset")
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["next_sequence"], 0)
        self.assertEqual(self.send(session_id, 0).status_code, 200)

    def test_media_contract_rejects_non_one_second_wav(self) -> None:
        session_id = self.start()
        response = self.send(session_id, 0, wav_bytes(frames=8_000))
        self.assertEqual(response.status_code, 422)
        self.assertIn("exactly one second", response.json()["error"])

    def test_stop_removes_in_memory_session(self) -> None:
        session_id = self.start()
        self.assertEqual(self.client.delete(f"/v1/sessions/{session_id}").status_code, 204)
        self.assertEqual(self.send(session_id, 0).status_code, 404)

    def test_cors_is_an_explicit_allowlist(self) -> None:
        response = self.client.options(
            "/v1/sessions",
            headers={
                "Origin": "https://example.test",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers.get("access-control-allow-origin"), "https://example.test"
        )

    def test_frontend_production_contract_fixture(self) -> None:
        context = TestClient(
            create_app(settings(), engine=ProductionContractFixtureEngine())
        )
        with context as client:
            session = client.post("/v1/sessions", json={}).json()
            self.assertEqual(session["schema_version"], SCHEMA_VERSION)
            self.assertEqual(session["status"], "running")
            self.assertEqual(session["mode"], "languagebind")
            self.assertEqual(
                session["model"], "LanguageBind_unit (Audio_FT + Image)"
            )
            self.assertEqual(session["classes"], list(LLP_CLASSES))
            self.assertNotIn("ground_truth", session)
            response = client.post(
                f'/v1/sessions/{session["session_id"]}/chunks',
                files={
                    "sequence": (None, "0", "text/plain"),
                    "frame": ("frame.jpg", jpeg_bytes(), "image/jpeg"),
                    "audio": ("audio.wav", wav_bytes(), "audio/wav"),
                },
            ).json()
            self.assertEqual(response["schema_version"], SCHEMA_VERSION)
            self.assertEqual(response["classes"], list(LLP_CLASSES))
            self.assertNotIn("ground_truth", response)
            self.assertEqual(response["dense"]["score_kind"], "class_axis_zscore_sigmoid")
            self.assertEqual(
                response["scope"]["score_kind"], "raw_nonnegative_stage2_weight"
            )
            self.assertTrue(response["scope"]["causal"])
            self.assertEqual(response["scope"]["future_segments_used"], 0)
            self.assertEqual(response["scope"]["prefix_segments"], 1)


class SessionStoreTests(unittest.TestCase):
    def test_ttl_is_absolute_and_lookups_do_not_refresh_it(self) -> None:
        now = [100.0]
        store = SessionStore(
            state_factory=EngineState.empty,
            ttl_seconds=30,
            max_sessions=2,
            max_seconds=60,
            clock=lambda: now[0],
        )
        session = store.create()
        now[0] += 20.0
        self.assertIs(store.get(session.session_id), session)
        self.assertEqual(session.last_seen, 100.0)
        now[0] += 11.0
        with self.assertRaises(SessionNotFound):
            store.get(session.session_id)

    def test_reset_does_not_replenish_sixty_chunk_budget(self) -> None:
        store = SessionStore(
            state_factory=EngineState.empty,
            ttl_seconds=300,
            max_sessions=2,
            max_seconds=60,
        )
        session = store.create()
        session.next_sequence = 59
        session.accepted_chunks = 59
        store.validate_sequence(session, 59)
        store.commit(session, EngineState.empty())
        store.reset(session.session_id)
        self.assertEqual(session.next_sequence, 0)
        self.assertEqual(session.accepted_chunks, 60)
        with self.assertRaises(SequenceConflict):
            store.validate_sequence(session, 0)

    def test_purge_never_removes_an_inflight_locked_session(self) -> None:
        now = [100.0]
        store = SessionStore(
            state_factory=EngineState.empty,
            ttl_seconds=30,
            max_sessions=2,
            max_seconds=60,
            clock=lambda: now[0],
        )
        session = store.create()

        async def exercise() -> None:
            await session.lock.acquire()
            try:
                now[0] += 31.0
                self.assertEqual(store.active_count, 1)
            finally:
                session.lock.release()

        asyncio.run(exercise())
        self.assertEqual(store.active_count, 0)


class FailClosedConfigTests(unittest.TestCase):
    def test_mock_is_never_the_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configured = Settings.from_env()
        self.assertEqual(configured.mode, "languagebind")

    def test_production_does_not_fall_back_when_assets_are_missing(self) -> None:
        configured = settings()
        production = Settings(**{**configured.__dict__, "mode": "languagebind"})
        with self.assertRaisesRegex(RuntimeError, "no fallback"):
            production.validate_production()

    def test_frozen_asset_hashes_are_exact_file_hashes(self) -> None:
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.npy"
            payload = b"frozen-languagebind-asset"
            path.write_bytes(payload)
            self.assertEqual(file_sha256(path), hashlib.sha256(payload).hexdigest())
        self.assertEqual(
            FROZEN_ASSET_SHA256["SCOPE_LB_AUDIO_PROTOTYPES"],
            "02028e363033337d8fa2e465ec188a2693587aa066938e484d9857332d1275fd",
        )
        self.assertIn(
            "models--LanguageBind--LanguageBind_Audio_FT/snapshots/"
            "4820c496563c46acfb1ff9a486fae5319f16257e/pytorch_model.bin",
            FROZEN_CHECKPOINT_SHA256,
        )

    def test_tree_manifest_is_deterministic_and_ignores_bytecode_cache(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package").mkdir()
            (root / "package" / "a.py").write_text("a = 1\n", encoding="utf-8")
            first = tree_manifest_sha256(root)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "a.pyc").write_bytes(b"runtime cache")
            self.assertEqual(tree_manifest_sha256(root), first)
            (root / "package" / "a.py").write_text("a = 2\n", encoding="utf-8")
            self.assertNotEqual(tree_manifest_sha256(root), first)

    def test_frozen_gpu_prediction_artifact_hash(self) -> None:
        path = BACKEND_ROOT / "audit" / "golden_F0Omj8D7rOg_seq0_prediction.json"
        self.assertEqual(
            file_sha256(path),
            "07f99bfa9c0b7a738c463813a934eb8df0a2f8ba88f5870ccd883ac7bbd90e82",
        )
        self.assertEqual(FROZEN_RUNTIME_VERSIONS["python"], "3.10.19")
        self.assertEqual(FROZEN_RUNTIME_VERSIONS["torch"], "2.9.1+cu128")
        self.assertEqual(FROZEN_RUNTIME_VERSIONS["torchaudio"], "2.9.1+cu128")
        synthetic = (
            BACKEND_ROOT / "audit" / "synthetic_constant_rgb_zero_audio_prediction.json"
        )
        self.assertEqual(file_sha256(synthetic), SYNTHETIC_GOLDEN_SHA256)

    def test_public_final_method_modules_are_loaded_and_executable_without_gpu(self) -> None:
        scope_repo = Path(
            os.environ.get(
                "SCOPE_REPO_PATH", str(BACKEND_ROOT.parents[2] / "SCoPE-main")
            )
        ).resolve()
        _validate_scope_source(scope_repo)
        final_core, streaming_core = _load_method_modules(scope_repo)
        import numpy as np

        sparse = np.zeros((1, 2, 25), dtype=np.float32)
        sparse[0, 0, 3] = 1.0
        sparse[0, 1, 3] = 0.5
        quality = np.asarray([[1.0, 0.5]], dtype=np.float32)
        prior = final_core.reliability_pooled_prior(sparse, quality)
        self.assertAlmostEqual(float(prior[0, 3]), 0.625)
        penalty = final_core.classwise_fixed_mean_penalty(prior, eta_target=16.0)
        self.assertAlmostEqual(float(penalty.mean()), 0.3, places=6)
        weights_a = np.zeros((1, 1, 25), dtype=np.float32)
        weights_v = np.zeros((1, 1, 25), dtype=np.float32)
        weights_a[0, 0, [2, 3]] = [0.8, 0.2]
        weights_v[0, 0, [2, 3]] = [0.7, 0.1]
        decoded = streaming_core.scope_zero_latency_raw(weights_a, weights_v)
        self.assertTrue(bool(decoded["audio_visual"][0, 0, 2]))

    def test_canonical_languagebind_vocab_storage_is_transposed(self) -> None:
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prototype.npy"
            stored = np.arange(768 * 25, dtype=np.float32).reshape(768, 25)
            np.save(path, stored)
            loaded = _load_prototypes(path, "test prototypes")
        self.assertEqual(loaded.shape, (25, 768))
        np.testing.assert_array_equal(loaded, stored.T)


if __name__ == "__main__":
    unittest.main()
