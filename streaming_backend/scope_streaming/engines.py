from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.machinery
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import types
from typing import Any

import numpy as np

from .config import Settings, file_sha256
from .contract import BRANCHES, DENSE_THRESHOLD, LLP_CLASSES, NUM_CLASSES
from .media import DecodedChunk


FROZEN_SCOPE_SOURCE_SHA256 = {
    "avvp_stage12/constants.py": (
        "58b5582e6a7e34105716b7ff76976bbc77fae523058ff03eed0b9cc6936c4941"
    ),
    "avvp_stage12/metrics.py": (
        "582c9d62e9f68e47a5bbccdf84300b8525bf15d7e22cff834c97beab779e49cd"
    ),
    "avvp_stage12/pipeline.py": (
        "91853097e96708253b2b7d13ee9ebbcf7ce9c2b0f1e61e5cd7fcceaaf2d5b9fa"
    ),
    "avvp_stage12/solver.py": (
        "49e59933a7c24c4d149c3ac0a851b4860ca4e5ca0a81545de1de3cef97929d0c"
    ),
}

FROZEN_METHOD_SOURCE_SHA256 = {
    "final_method_core.py": (
        "7d033376d9d6eb916620f1782c4cef1a090197e128335df7d2a445e40814aabd"
    ),
    "streaming_demo_core.py": (
        "7eaa426e107501663dd2fcf046e2078d4ba222bb8024d86861308abff553f782"
    ),
    "final_method_lock.json": (
        "bac40fa676116de56ecfa4a5ed35617f9dc3e2d5eb55fb0028a801fef1b60afc"
    ),
}

FROZEN_RUNTIME_VERSIONS = {
    "python": "3.10.19",
    "numpy": "1.26.4",
    "torch": "2.9.1+cu128",
    "torchaudio": "2.9.1+cu128",
    "torchvision": "0.24.1",
    "Pillow": "12.2.0",
}

SYNTHETIC_GOLDEN_FILENAME = "synthetic_constant_rgb_zero_audio_prediction.json"
SYNTHETIC_GOLDEN_SHA256 = (
    "95abcf9024c4b7012856d021769c8e1a8ff77ae50f980dd40b26dc7ea0e90fb6"
)
SYNTHETIC_IMAGE_RGB = (30, 120, 220)
SYNTHETIC_IMAGE_SHAPE = (48, 64, 3)
SYNTHETIC_IMAGE_SHA256 = (
    "193c9a93761c1943a1711b5795d87187593cbebdf3c18ed479ca2607c7b6d311"
)
SYNTHETIC_AUDIO_SHA256 = (
    "4f7988030a00d082fe445e00a2ac5dab502300ff1b80e8592dd569867b60ef74"
)


def _require_exact_file(path: Path, expected: str, name: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"frozen {name} is missing: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"frozen {name} SHA-256 mismatch: expected {expected}, observed {observed}"
        )


def _validate_scope_source(scope_repo: Path) -> None:
    for relative, expected in FROZEN_SCOPE_SOURCE_SHA256.items():
        _require_exact_file(scope_repo / relative, expected, relative)


def _validate_runtime_versions(torch_module: Any, torchaudio_module: Any) -> None:
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch_module.__version__),
        "torchaudio": str(torchaudio_module.__version__),
        "torchvision": importlib_metadata.version("torchvision"),
        "Pillow": importlib_metadata.version("Pillow"),
    }
    mismatches = [
        f"{name} expected {FROZEN_RUNTIME_VERSIONS[name]}, observed {value}"
        for name, value in observed.items()
        if value != FROZEN_RUNTIME_VERSIONS[name]
    ]
    if mismatches:
        raise RuntimeError("frozen runtime version mismatch: " + "; ".join(mismatches))


@dataclass(frozen=True)
class EngineState:
    sparse_audio: np.ndarray
    sparse_visual: np.ndarray
    quality_audio: np.ndarray
    quality_visual: np.ndarray

    @classmethod
    def empty(cls) -> "EngineState":
        return cls(
            sparse_audio=np.empty((0, NUM_CLASSES), dtype=np.float32),
            sparse_visual=np.empty((0, NUM_CLASSES), dtype=np.float32),
            quality_audio=np.empty((0,), dtype=np.float32),
            quality_visual=np.empty((0,), dtype=np.float32),
        )

    @property
    def length(self) -> int:
        return int(self.sparse_audio.shape[0])


def _branch(scores: np.ndarray, prediction: np.ndarray) -> dict[str, list[float] | list[bool]]:
    score_row = np.asarray(scores, dtype=np.float32).reshape(NUM_CLASSES)
    pred_row = np.asarray(prediction, dtype=bool).reshape(NUM_CLASSES)
    if not np.isfinite(score_row).all():
        raise RuntimeError("non-finite prediction scores")
    return {
        "scores": [float(value) for value in score_row],
        "predictions": [bool(value) for value in pred_row],
    }


class MockEngine:
    """Deterministic UI test double.  It is never selected implicitly."""

    mode = "mock"
    model_name = "explicit-deterministic-mock"

    def new_state(self) -> EngineState:
        return EngineState.empty()

    def predict(
        self, state: EngineState, media: DecodedChunk
    ) -> tuple[dict[str, Any], EngineState, dict[str, float]]:
        start = time.perf_counter()
        image_size = getattr(media.image, "size", (0, 0))
        signature = (
            f"{state.length}:{image_size}:{float(media.waveform.mean()):.8f}:"
            f"{float(media.waveform.std()):.8f}"
        ).encode("ascii")
        seed = int.from_bytes(hashlib.sha256(signature).digest()[:8], "big")
        indices = np.arange(NUM_CLASSES, dtype=np.float32)
        phase = float(seed % 10_000) / 997.0
        dense_a = ((np.sin(indices * 1.71 + phase) + 1.0) * 0.5).astype(np.float32)
        dense_v = ((np.cos(indices * 1.37 + phase * 0.73) + 1.0) * 0.5).astype(np.float32)
        dense_av = np.minimum(dense_a, dense_v)
        scope_a = np.maximum(dense_a - 0.42, 0.0)
        scope_v = np.maximum(dense_v - 0.42, 0.0)
        common = (scope_a > 0.0) & (scope_v > 0.0)
        scope_av = np.where(common, 0.45 * scope_a + 0.55 * scope_v, 0.0)

        def gap_prediction(values: np.ndarray) -> np.ndarray:
            prediction = np.zeros(NUM_CLASSES, dtype=bool)
            active = np.flatnonzero(values > 1.0e-6)
            if not active.size:
                return prediction
            ranked = active[np.argsort(-values[active], kind="stable")]
            ranked_values = values[ranked]
            gaps = ranked_values - np.concatenate((ranked_values[1:], np.zeros(1)))
            prediction[ranked[: int(np.argmax(gaps)) + 1]] = True
            return prediction

        next_state = EngineState(
            sparse_audio=np.concatenate((state.sparse_audio, scope_a[None, :]), axis=0),
            sparse_visual=np.concatenate((state.sparse_visual, scope_v[None, :]), axis=0),
            quality_audio=np.concatenate((state.quality_audio, np.ones(1, dtype=np.float32))),
            quality_visual=np.concatenate((state.quality_visual, np.ones(1, dtype=np.float32))),
        )
        output = {
            "dense": {
                "threshold": DENSE_THRESHOLD,
                "comparator": ">",
                "score_kind": "mock_continuous_score",
                "branches": {
                    "audio": _branch(dense_a, dense_a > DENSE_THRESHOLD),
                    "visual": _branch(dense_v, dense_v > DENSE_THRESHOLD),
                    "audio_visual": _branch(dense_av, dense_av > DENSE_THRESHOLD),
                },
            },
            "scope": {
                "causal": True,
                "future_segments_used": 0,
                "score_kind": "mock_sparse_weight",
                "branches": {
                    "audio": _branch(scope_a, gap_prediction(scope_a)),
                    "visual": _branch(scope_v, gap_prediction(scope_v)),
                    "audio_visual": _branch(scope_av, gap_prediction(scope_av)),
                },
            },
        }
        elapsed = (time.perf_counter() - start) * 1000.0
        return output, next_state, {"encode": 0.0, "inference": elapsed}


def _load_method_modules(scope_repo: Path) -> tuple[Any, Any]:
    relative = Path("scope_iclr_final_paper_campaign_20260806")
    candidates = [scope_repo / "experiments" / relative, scope_repo / "scripts" / relative]
    required_files = ("final_method_core.py", "streaming_demo_core.py")
    directory = next(
        (
            candidate
            for candidate in candidates
            if all((candidate / filename).is_file() for filename in required_files)
        ),
        None,
    )
    if directory is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(
            "public SCoPE final method modules are missing; searched: " + searched
        )
    for filename, expected in FROZEN_METHOD_SOURCE_SHA256.items():
        _require_exact_file(directory / filename, expected, filename)
    package_name = "_scope_live_public_method"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    loaded: dict[str, Any] = {}
    for name in ("final_method_core", "streaming_demo_core"):
        path = directory / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not import public SCoPE module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    loaded["final_method_core"].load_method_lock()
    return loaded["final_method_core"], loaded["streaming_demo_core"]


def _load_array(path: Path, name: str) -> np.ndarray:
    array = np.load(path, allow_pickle=False).astype(np.float32)
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains non-finite values: {path}")
    return array


def _load_prototypes(path: Path, name: str) -> np.ndarray:
    """Load either the stored (D,25) vocab matrix or an adapter-ready (25,D) copy.

    The canonical ``v25_languagebind_{clap,clip}.npy`` files follow the public
    SCoPE vocabulary loader's on-disk convention and therefore need the same
    transpose performed by ``load_prompt_vocab``.
    """

    array = _load_array(path, name)
    if array.ndim != 2:
        raise RuntimeError(f"{name} must be a two-dimensional matrix")
    if array.shape[0] == NUM_CLASSES:
        return array
    if array.shape[1] == NUM_CLASSES:
        return array.T.copy()
    raise RuntimeError(
        f"{name} must have one class axis of length {NUM_CLASSES}, got {array.shape}"
    )


def _install_noop_wandb_stub() -> None:
    """Keep the inference-only pinned HF stack independent of broken W&B installs.

    The vendored Accelerate version imports W&B eagerly when the package is
    discoverable, although this service never constructs a tracker.  Some of
    the frozen server environments contain an incompatible W&B/protobuf pair;
    a tiny importable stub makes that optional integration unavailable without
    changing LanguageBind, PEFT, Accelerate, or model inference.
    """

    for module_name in tuple(sys.modules):
        if module_name == "wandb" or module_name.startswith("wandb."):
            del sys.modules[module_name]
    stub = types.ModuleType("wandb")
    stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    sys.modules["wandb"] = stub


class LanguageBindScopeEngine:
    """Persistent LanguageBind encoder plus the public final SCoPE primitives."""

    mode = "languagebind"
    model_name = "LanguageBind_unit (Audio_FT + Image)"

    def __init__(self, settings: Settings) -> None:
        settings.validate_production()
        self.settings = settings
        self._assert_exact_solver_environment()
        _validate_scope_source(settings.scope_repo)
        self.final_core, self.streaming_core = _load_method_modules(settings.scope_repo)
        self.final_core.assert_adapter_source_safe(Path(__file__).resolve())
        if str(settings.scope_repo) not in sys.path:
            sys.path.insert(0, str(settings.scope_repo))
        self.pipeline = importlib.import_module("avvp_stage12.pipeline")
        self.metrics = importlib.import_module("avvp_stage12.metrics")
        pipeline_file = Path(self.pipeline.__file__).resolve()
        if settings.scope_repo not in pipeline_file.parents:
            raise RuntimeError(f"imported avvp_stage12 outside configured SCoPE repo: {pipeline_file}")
        imported_classes = tuple(importlib.import_module("avvp_stage12.constants").LLP_CATS)
        if imported_classes != LLP_CLASSES:
            raise RuntimeError("configured SCoPE repository does not expose the canonical LLP-25 order")

        assert settings.audio_prototypes is not None
        assert settings.visual_prototypes is not None
        assert settings.audio_mean is not None
        assert settings.visual_mean is not None
        self.audio_prototypes = _load_prototypes(
            settings.audio_prototypes, "audio prototypes"
        )
        self.visual_prototypes = _load_prototypes(
            settings.visual_prototypes, "visual prototypes"
        )
        self.audio_mean = _load_array(settings.audio_mean, "audio mean").reshape(-1)
        self.visual_mean = _load_array(settings.visual_mean, "visual mean").reshape(-1)
        if self.audio_prototypes.ndim != 2 or self.audio_prototypes.shape[0] != NUM_CLASSES:
            raise RuntimeError("audio prototypes must have shape (25,D)")
        if self.visual_prototypes.shape != self.audio_prototypes.shape:
            raise RuntimeError("LanguageBind audio/visual prototype shapes must match")
        dimension = self.audio_prototypes.shape[1]
        if self.audio_mean.shape != (dimension,) or self.visual_mean.shape != (dimension,):
            raise RuntimeError("LanguageBind external means must have shape (D,)")

        for optional in (settings.languagebind_vendor, settings.languagebind_code):
            if optional is not None and str(optional) not in sys.path:
                sys.path.insert(0, str(optional))
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        _install_noop_wandb_stub()
        import torch
        import torch.nn.functional as torch_functional
        import torchaudio

        # The frozen LanguageBind/PyTorchVideo stack imports torchvision's
        # historical public module name.  New torchvision releases retain the
        # implementation under the private name only; expose the same
        # compatibility alias used by the verified cache extractor.
        import torchvision.transforms._functional_tensor as functional_tensor

        sys.modules.setdefault(
            "torchvision.transforms.functional_tensor", functional_tensor
        )
        _validate_runtime_versions(torch, torchaudio)

        if settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"{settings.device} requested but CUDA is unavailable")
        languagebind_module = importlib.import_module("languagebind")
        if settings.languagebind_code is not None:
            module_file = Path(languagebind_module.__file__).resolve()
            if settings.languagebind_code not in module_file.parents:
                raise RuntimeError(
                    "imported LanguageBind outside configured LANGUAGEBIND_CODE_PATH: "
                    f"{module_file}"
                )
        LanguageBind = languagebind_module.LanguageBind
        transform_dict = languagebind_module.transform_dict

        assert settings.languagebind_cache is not None
        self.torch = torch
        self.torch_functional = torch_functional
        self.device = torch.device(settings.device)
        clip_type = {
            "audio": "LanguageBind_Audio_FT",
            "image": "LanguageBind_Image",
        }
        self.model = LanguageBind(
            clip_type=clip_type,
            use_temp=False,
            cache_dir=str(settings.languagebind_cache),
        ).to(self.device).eval()
        self.image_transform = transform_dict["image"](
            self.model.modality_config["image"]
        ).transform
        self.audio_transform = transform_dict["audio"](
            self.model.modality_config["audio"]
        ).transform
        self._validate_synthetic_golden()

    @staticmethod
    def _assert_exact_solver_environment() -> None:
        forbidden = {
            "SCOPE_USE_INNER_L2": "0",
            "SCOPE_USE_OUTER_L2": "0",
            "SCOPE_ZERO_MEAN": "1",
        }
        hits = [name for name, value in forbidden.items() if os.environ.get(name) == value]
        if float(os.environ.get("SCOPE_ELASTIC_NET_L2", "0") or "0") != 0.0:
            hits.append("SCOPE_ELASTIC_NET_L2")
        if os.environ.get("SCOPE_ELASTIC_NET_L2_FILE"):
            hits.append("SCOPE_ELASTIC_NET_L2_FILE")
        sparse_mode = os.environ.get("SCOPE_SPARSE_CONF_NORM", "max_c")
        if sparse_mode != "max_c":
            hits.append("SCOPE_SPARSE_CONF_NORM")
        reliability_mode = os.environ.get("SCOPE_RELIABILITY_CLIP", "01")
        if reliability_mode != "01":
            hits.append("SCOPE_RELIABILITY_CLIP")
        if hits:
            raise RuntimeError("non-paper solver environment is active: " + ", ".join(hits))

    def new_state(self) -> EngineState:
        return EngineState.empty()

    def _validate_synthetic_golden(self) -> None:
        from PIL import Image

        golden_path = (
            Path(__file__).resolve().parents[1] / "audit" / SYNTHETIC_GOLDEN_FILENAME
        )
        _require_exact_file(
            golden_path, SYNTHETIC_GOLDEN_SHA256, "synthetic GPU golden artifact"
        )
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        if golden.get("schema") != "scope-streaming-synthetic-golden-prediction-v1":
            raise RuntimeError("synthetic GPU golden artifact has the wrong schema")

        image_array = np.empty(SYNTHETIC_IMAGE_SHAPE, dtype=np.uint8)
        image_array[...] = np.asarray(SYNTHETIC_IMAGE_RGB, dtype=np.uint8)
        waveform = np.zeros((16_000,), dtype=np.float32)
        image_hash = hashlib.sha256(
            np.ascontiguousarray(image_array).tobytes(order="C")
        ).hexdigest()
        audio_hash = hashlib.sha256(
            waveform.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest()
        if image_hash != SYNTHETIC_IMAGE_SHA256 or audio_hash != SYNTHETIC_AUDIO_SHA256:
            raise RuntimeError("synthetic GPU golden input construction changed")

        output, next_state, _timings = self.predict(
            self.new_state(),
            DecodedChunk(
                image=Image.fromarray(image_array, mode="RGB"), waveform=waveform
            ),
        )
        if next_state.length != 1 or output != golden.get("output"):
            raise RuntimeError(
                "synthetic GPU golden inference mismatch; refusing to start production"
            )

    def _encode(self, media: DecodedChunk) -> tuple[np.ndarray, np.ndarray]:
        image_tensor = self.image_transform(media.image).unsqueeze(0).to(self.device)
        waveform = self.torch.from_numpy(media.waveform.copy()).unsqueeze(0)
        audio_tensor = self.audio_transform((waveform, 16_000)).unsqueeze(0).to(self.device)
        inputs = {
            "image": {"pixel_values": image_tensor},
            "audio": {"pixel_values": audio_tensor},
        }
        with self.torch.inference_mode():
            embeddings = self.model(inputs)
            audio = self.torch_functional.normalize(
                embeddings["audio"].float(), p=2, dim=-1
            )
            visual = self.torch_functional.normalize(
                embeddings["image"].float(), p=2, dim=-1
            )
        audio_np = audio.detach().cpu().numpy().reshape(-1).astype(np.float32)
        visual_np = visual.detach().cpu().numpy().reshape(-1).astype(np.float32)
        expected = self.audio_prototypes.shape[1]
        if audio_np.shape != (expected,) or visual_np.shape != (expected,):
            raise RuntimeError(
                f"LanguageBind embedding dimension mismatch: {audio_np.shape}/{visual_np.shape}"
            )
        return audio_np, visual_np

    def _prepare_current(
        self, name: str, feature: np.ndarray, prototypes: np.ndarray, mean: np.ndarray
    ) -> Any:
        # prepare_modality requires a video-shaped vector, but every executed
        # call below is segment-only (run_segment_decomposition,
        # compute_dense_similarity, run_weighted_segment_decomposition).  The
        # current-prefix vector is supplied only to satisfy that shape contract;
        # run_stage12/run_video_decomposition are deliberately never invoked.
        return self.pipeline.prepare_modality(
            name,
            feature.reshape(1, 1, -1),
            feature.reshape(1, -1),
            prototypes,
            segment_mean_override=mean,
            video_mean_override=mean,
        )

    @staticmethod
    def _append_rows(existing: np.ndarray, row: np.ndarray) -> np.ndarray:
        return np.concatenate((existing, row.reshape(1, -1).astype(np.float32)), axis=0)

    def predict(
        self, state: EngineState, media: DecodedChunk
    ) -> tuple[dict[str, Any], EngineState, dict[str, float]]:
        encode_start = time.perf_counter()
        audio_feature, visual_feature = self._encode(media)
        encode_ms = (time.perf_counter() - encode_start) * 1000.0

        inference_start = time.perf_counter()
        audio_modality = self._prepare_current(
            "audio", audio_feature, self.audio_prototypes, self.audio_mean
        )
        visual_modality = self._prepare_current(
            "visual", visual_feature, self.visual_prototypes, self.visual_mean
        )
        stage1_a = self.pipeline.run_segment_decomposition(
            audio_modality, self.final_core.LAMBDA0, 200, self.settings.device
        )
        stage1_v = self.pipeline.run_segment_decomposition(
            visual_modality, self.final_core.LAMBDA0, 200, self.settings.device
        )

        sparse_a, quality_a, _ = self.pipeline.compute_reliable_sparse_confidence(
            stage1_a["weights"], stage1_a["recon_center"]
        )
        sparse_v, quality_v, _ = self.pipeline.compute_reliable_sparse_confidence(
            stage1_v["weights"], stage1_v["recon_center"]
        )
        next_state = EngineState(
            sparse_audio=self._append_rows(state.sparse_audio, sparse_a[0, 0]),
            sparse_visual=self._append_rows(state.sparse_visual, sparse_v[0, 0]),
            quality_audio=np.concatenate(
                (state.quality_audio, np.asarray([quality_a[0, 0]], dtype=np.float32))
            ),
            quality_visual=np.concatenate(
                (state.quality_visual, np.asarray([quality_v[0, 0]], dtype=np.float32))
            ),
        )
        # This is the final public P=mean(q*s) implementation evaluated on the
        # received prefix only.  No global audio vector or future segment exists.
        prior_a = self.final_core.reliability_pooled_prior(
            next_state.sparse_audio[None, :, :], next_state.quality_audio[None, :]
        )[0]
        prior_v = self.final_core.reliability_pooled_prior(
            next_state.sparse_visual[None, :, :], next_state.quality_visual[None, :]
        )[0]
        penalty_a = self.final_core.classwise_fixed_mean_penalty(
            prior_v[None, :],
            eta_target=self.final_core.ETA_V_TO_A,
            lambda0=self.final_core.LAMBDA0,
        )
        penalty_v = self.final_core.classwise_fixed_mean_penalty(
            prior_a[None, :],
            eta_target=self.final_core.ETA_A_TO_V,
            lambda0=self.final_core.LAMBDA0,
        )
        stage2_a = self.pipeline.run_weighted_segment_decomposition(
            audio_modality, penalty_a, 200, self.settings.device
        )["weights"]
        stage2_v = self.pipeline.run_weighted_segment_decomposition(
            visual_modality, penalty_v, 200, self.settings.device
        )["weights"]
        scope_predictions = self.streaming_core.scope_zero_latency_raw(stage2_a, stage2_v)
        support_eps = float(self.final_core.SUPPORT_EPS)
        fusion_alpha = float(self.final_core.FUSION_ALPHA_AUDIO)
        common_support = (stage2_a > support_eps) & (stage2_v > support_eps)
        stage2_av = np.where(
            common_support,
            fusion_alpha * stage2_a + (1.0 - fusion_alpha) * stage2_v,
            0.0,
        ).astype(np.float32)

        dense_a = self.metrics.norm_similarities_np(
            self.pipeline.compute_dense_similarity(audio_modality), exclude_zero=False
        )[0, 0]
        dense_v = self.metrics.norm_similarities_np(
            self.pipeline.compute_dense_similarity(visual_modality), exclude_zero=False
        )[0, 0]
        dense_av = np.minimum(dense_a, dense_v)
        dense_predictions = {
            "audio": dense_a > DENSE_THRESHOLD,
            "visual": dense_v > DENSE_THRESHOLD,
            "audio_visual": dense_av > DENSE_THRESHOLD,
        }
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        output = {
            "dense": {
                "threshold": DENSE_THRESHOLD,
                "comparator": ">",
                "score_kind": "class_axis_zscore_sigmoid",
                "branches": {
                    "audio": _branch(dense_a, dense_predictions["audio"]),
                    "visual": _branch(dense_v, dense_predictions["visual"]),
                    "audio_visual": _branch(dense_av, dense_predictions["audio_visual"]),
                },
            },
            "scope": {
                "causal": True,
                "future_segments_used": 0,
                "score_kind": "raw_nonnegative_stage2_weight",
                "prefix_segments": next_state.length,
                "branches": {
                    "audio": _branch(stage2_a[0, 0], scope_predictions["audio"][0, 0]),
                    "visual": _branch(stage2_v[0, 0], scope_predictions["visual"][0, 0]),
                    "audio_visual": _branch(
                        stage2_av[0, 0], scope_predictions["audio_visual"][0, 0]
                    ),
                },
            },
        }
        if set(output["dense"]["branches"]) != set(BRANCHES):
            raise AssertionError("Dense branch contract changed")
        return output, next_state, {"encode": encode_ms, "inference": inference_ms}


def build_engine(settings: Settings) -> MockEngine | LanguageBindScopeEngine:
    if settings.mode == "mock":
        return MockEngine()
    return LanguageBindScopeEngine(settings)
