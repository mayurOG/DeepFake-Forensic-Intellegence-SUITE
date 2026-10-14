"""Central configuration.

Everything is overridable through environment variables (or a .env file) so
the same code runs on a laptop CPU and on a GPU box without edits.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


@dataclass(frozen=True)
class Settings:
    # Pretrained image classifier used as the default Tier-1 backbone.
    image_model_id: str = field(default_factory=lambda: _env( "IMAGE_MODEL_ID", "prithivMLmods/deepfake-detector-model-v1" ))
    # Optional frozen foundation encoder (e.g. a DINOv2/DINOv3, SigLIP2 or
    # PE-CLIP checkpoint) used together with a trained linear probe. Leave
    # empty to rely on the classifier above. See README, "Training".
    embed_model_id: str = field(default_factory=lambda: _env("EMBED_MODEL_ID", ""))
    audio_model_id: str = field(default_factory=lambda: _env( "AUDIO_MODEL_ID", "mo-thecreator/Deepfake-audio-detection" ))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN", ""))
    # Directory holding trained/fitted artifacts (probe, calibration, ...).
    artifact_dir: str = field(default_factory=lambda: _env("DEEPFAKE_ARTIFACT_DIR", "artifacts"))
    # "auto" | "cpu" | "cuda" | "mps"
    device: str = field(default_factory=lambda: _env("DEEPFAKE_DEVICE", "auto"))
    # Conformal risk level: the true label is inside the predicted set with
    # probability >= 1 - alpha (given exchangeable calibration data).
    alpha: float = field(default_factory=lambda: float(_env("DEEPFAKE_ALPHA", "0.1")))
    # Number of frames in the cheap Tier-1 video screen.
    tier1_frames: int = field(default_factory=lambda: int(_env("DEEPFAKE_TIER1_FRAMES", "12")))
    # Maximum seconds of video used by the Tier-2 face-signal experts.
    face_signal_seconds: float = field(default_factory=lambda: float(_env("DEEPFAKE_FACE_SECONDS", "8")))


def get_settings() -> Settings:
    return Settings()


def artifact_path(name: str) -> str:
    return os.path.join(get_settings().artifact_dir, name)
