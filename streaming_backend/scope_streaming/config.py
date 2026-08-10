from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path


FROZEN_ASSET_SHA256 = {
    "SCOPE_LB_AUDIO_PROTOTYPES": (
        "02028e363033337d8fa2e465ec188a2693587aa066938e484d9857332d1275fd"
    ),
    "SCOPE_LB_VISUAL_PROTOTYPES": (
        "02028e363033337d8fa2e465ec188a2693587aa066938e484d9857332d1275fd"
    ),
    "SCOPE_LB_AUDIO_MEAN": (
        "ebb9b66d005bd1b0b9ed9f470f872587d3a0f5e8a6590abae4d03f19e9f42ae2"
    ),
    "SCOPE_LB_VISUAL_MEAN": (
        "256f3f934e5cf26191b2418a0c5d0203b326198be68bb00fb32dbbdf9eb7690f"
    ),
}

FROZEN_CHECKPOINT_SHA256 = {
    (
        "models--LanguageBind--LanguageBind_Audio_FT/snapshots/"
        "4820c496563c46acfb1ff9a486fae5319f16257e/pytorch_model.bin"
    ): "b0fbb6a2703c1021d754262b5dde79e69f83b0b53763d327366e4d93efa15253",
    (
        "models--LanguageBind--LanguageBind_Audio_FT/snapshots/"
        "4820c496563c46acfb1ff9a486fae5319f16257e/config.json"
    ): "69f84d3dc45c2938ea61e7ebfe01e8780fb5a5a0198d0507961904fd534fec04",
    (
        "models--LanguageBind--LanguageBind_Image/snapshots/"
        "d8c2e37b439f4fc47c649dc8b90cdcd3a4e0c80e/pytorch_model.bin"
    ): "99c9382819ef4021e9b2600f030f267d18cc4d5dad39928fdd83ed48887e94bd",
    (
        "models--LanguageBind--LanguageBind_Image/snapshots/"
        "d8c2e37b439f4fc47c649dc8b90cdcd3a4e0c80e/config.json"
    ): "04b4eb8bf1c69372fc74c0a8799c4850898f8fe3c57a2da01eddae240bc9aeac",
}

FROZEN_TREE_MANIFEST_SHA256 = {
    "LANGUAGEBIND_CODE_PATH": (
        "5d4c0ad415088fa4abbbce607548b0937357e0fea4cdd7dc5cae2be58aa623aa"
    ),
    "LANGUAGEBIND_VENDOR_PATH": (
        "fa109abc471058bd4ce91ae119d8c244310fd2832dd1baa9b412d32ddb6855e4"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest_sha256(root: Path) -> str:
    """Match the frozen ``find | sort | sha256sum`` source manifest.

    Runtime bytecode caches are deliberately excluded. Symlinks are excluded
    just as ``find . -type f`` excludes them; checkpoint symlinks are pinned
    separately by hashing their resolved content.
    """

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    paths.sort(key=lambda path: path.relative_to(root).as_posix().encode("utf-8"))
    manifest = hashlib.sha256()
    for path in paths:
        relative = "./" + path.relative_to(root).as_posix()
        manifest.update(f"{file_sha256(path)}  {relative}\n".encode("utf-8"))
    return manifest.hexdigest()


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _optional_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


@dataclass(frozen=True)
class Settings:
    """Fail-closed runtime settings.

    Production is the default.  The deterministic mock is available only when
    ``SCOPE_STREAMING_MODE=mock`` is set explicitly.
    """

    mode: str
    scope_repo: Path
    languagebind_code: Path | None
    languagebind_vendor: Path | None
    languagebind_cache: Path | None
    audio_prototypes: Path | None
    visual_prototypes: Path | None
    audio_mean: Path | None
    visual_mean: Path | None
    device: str
    cors_origins: tuple[str, ...]
    session_ttl_seconds: int
    max_sessions: int
    max_seconds: int = 60
    max_frame_bytes: int = 700_000
    max_audio_bytes: int = 70_000
    max_request_bytes: int = 790_000
    max_frame_pixels: int = 2_073_600

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[4]
        mode = os.environ.get("SCOPE_STREAMING_MODE", "languagebind").strip().lower()
        if mode not in {"languagebind", "mock"}:
            raise ValueError("SCOPE_STREAMING_MODE must be 'languagebind' or explicit 'mock'")
        origins_raw = os.environ.get(
            "SCOPE_CORS_ORIGINS",
            "https://jaemojeong.github.io,http://localhost:8765,http://127.0.0.1:8765",
        )
        origins = tuple(item.strip().rstrip("/") for item in origins_raw.split(",") if item.strip())
        if not origins or "*" in origins:
            raise ValueError("SCOPE_CORS_ORIGINS must be a non-empty explicit allowlist")
        device = os.environ.get("SCOPE_DEVICE", "cuda:0").strip()
        if device == "cuda:3":
            raise ValueError("cuda:3 is reserved and cannot be used by this service")
        return cls(
            mode=mode,
            scope_repo=Path(
                os.environ.get("SCOPE_REPO_PATH", str(project_root / "SCoPE-main"))
            ).expanduser().resolve(),
            languagebind_code=_optional_path("LANGUAGEBIND_CODE_PATH"),
            languagebind_vendor=_optional_path("LANGUAGEBIND_VENDOR_PATH"),
            languagebind_cache=_optional_path("LANGUAGEBIND_CACHE_DIR"),
            audio_prototypes=_optional_path("SCOPE_LB_AUDIO_PROTOTYPES"),
            visual_prototypes=_optional_path("SCOPE_LB_VISUAL_PROTOTYPES"),
            audio_mean=_optional_path("SCOPE_LB_AUDIO_MEAN"),
            visual_mean=_optional_path("SCOPE_LB_VISUAL_MEAN"),
            device=device,
            cors_origins=origins,
            session_ttl_seconds=_env_int(
                "SCOPE_SESSION_TTL_SECONDS", 300, minimum=30, maximum=3600
            ),
            max_sessions=_env_int("SCOPE_MAX_SESSIONS", 16, minimum=1, maximum=128),
        )

    def validate_production(self) -> None:
        if self.mode != "languagebind":
            raise ValueError("production validation called outside languagebind mode")
        required = {
            "SCOPE_REPO_PATH": self.scope_repo,
            "LANGUAGEBIND_CODE_PATH": self.languagebind_code,
            "LANGUAGEBIND_VENDOR_PATH": self.languagebind_vendor,
            "LANGUAGEBIND_CACHE_DIR": self.languagebind_cache,
            "SCOPE_LB_AUDIO_PROTOTYPES": self.audio_prototypes,
            "SCOPE_LB_VISUAL_PROTOTYPES": self.visual_prototypes,
            "SCOPE_LB_AUDIO_MEAN": self.audio_mean,
            "SCOPE_LB_VISUAL_MEAN": self.visual_mean,
        }
        missing = [name for name, path in required.items() if path is None or not path.exists()]
        if missing:
            raise RuntimeError(
                "LanguageBind mode has no fallback; configure existing frozen assets: "
                + ", ".join(missing)
            )
        assert self.languagebind_vendor is not None
        if not (self.languagebind_vendor / "transformers-4.30.2.dist-info").is_dir():
            raise RuntimeError(
                "LANGUAGEBIND_VENDOR_PATH must contain the pinned Transformers 4.30.2 stack"
            )
        frozen_paths = {
            "SCOPE_LB_AUDIO_PROTOTYPES": self.audio_prototypes,
            "SCOPE_LB_VISUAL_PROTOTYPES": self.visual_prototypes,
            "SCOPE_LB_AUDIO_MEAN": self.audio_mean,
            "SCOPE_LB_VISUAL_MEAN": self.visual_mean,
        }
        for name, path in frozen_paths.items():
            assert path is not None
            observed = file_sha256(path)
            expected = FROZEN_ASSET_SHA256[name]
            if observed != expected:
                raise RuntimeError(
                    f"{name} SHA-256 mismatch: expected {expected}, observed {observed}"
                )
        frozen_trees = {
            "LANGUAGEBIND_CODE_PATH": self.languagebind_code,
            "LANGUAGEBIND_VENDOR_PATH": self.languagebind_vendor,
        }
        for name, path in frozen_trees.items():
            assert path is not None
            observed = tree_manifest_sha256(path)
            expected = FROZEN_TREE_MANIFEST_SHA256[name]
            if observed != expected:
                raise RuntimeError(
                    f"{name} source manifest mismatch: expected {expected}, observed {observed}"
                )
        assert self.languagebind_cache is not None
        for relative, expected in FROZEN_CHECKPOINT_SHA256.items():
            path = self.languagebind_cache / relative
            if not path.is_file():
                raise RuntimeError(f"frozen LanguageBind checkpoint file is missing: {path}")
            observed = file_sha256(path)
            if observed != expected:
                raise RuntimeError(
                    f"LanguageBind checkpoint SHA-256 mismatch for {relative}: "
                    f"expected {expected}, observed {observed}"
                )
