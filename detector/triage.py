"""Stage 0 - ingest and triage.

Computes a content hash, a *quality profile* (how degraded is this media?)
and a provenance check. The quality profile later sets the reliability
weight of every evidence stream: a compression-destroyed clip should not be
allowed to speak as loudly as a clean one.
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

_STD_LUMA_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def estimate_jpeg_quality(img: Image.Image) -> Optional[float]:
    """Estimate the IJG quality factor from the luminance quantisation table
    (None for non-JPEG images). The ratio of table means is order-agnostic."""
    q = getattr(img, "quantization", None)
    if not q:
        return None
    table = np.asarray(list(q[min(q.keys())]), dtype=np.float64)
    if table.size != 64:
        return None
    scale = table.mean() / _STD_LUMA_Q.mean() * 100.0
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return float(np.clip(quality, 1.0, 100.0))


def detect_c2pa(data: bytes) -> Dict[str, Any]:
    """Heuristic C2PA / Content Credentials detector.

    This only finds the *presence* of a manifest (JUMBF box labelled c2pa).
    It does NOT verify the signature chain - use a C2PA SDK for that. The
    result is reported as provenance context, never as proof of authenticity.
    """
    present = (b"c2pa" in data) and (b"jumb" in data)
    return {
        "c2pa_manifest_detected": bool(present),
        "signature_verified": False,
        "note": (
            "Content Credentials manifest present; signature NOT verified here."
            if present else "No Content Credentials manifest found."
        ),
    }


@dataclass
class QualityProfile:
    media_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    jpeg_quality: Optional[float] = None
    blur_laplacian_var: Optional[float] = None
    bitrate_kbps: Optional[float] = None
    bits_per_pixel: Optional[float] = None
    sample_rate: Optional[int] = None
    snr_db: Optional[float] = None
    clipping: Optional[bool] = None
    compression: str = "unknown"  # low | medium | high | unknown
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def neural_reliability(self) -> float:
        """How much to trust a neural score given this degradation level.
        Detectors are known to lose accuracy under heavy compression."""
        return {"low": 1.0, "medium": 0.9, "high": 0.75}.get(self.compression, 0.9)


def image_quality(img: Image.Image, raw: Optional[bytes] = None) -> QualityProfile:
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    q = estimate_jpeg_quality(img)
    qp = QualityProfile(
        media_type="image", width=img.width, height=img.height,
        jpeg_quality=q,
        blur_laplacian_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )
    if q is not None:
        qp.compression = "high" if q < 60 else "medium" if q < 85 else "low"
        if q < 60:
            qp.notes.append(f"Heavy JPEG compression (est. quality {q:.0f}).")
    elif min(img.size) < 224:
        qp.compression = "high"
    else:
        qp.compression = "low"
    if min(img.size) < 224:
        qp.notes.append("Low resolution (<224 px): detector accuracy is reduced.")
    return qp


def video_quality(path: str) -> QualityProfile:
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    dur = float(n / fps) if n and n > 0 else None
    size = os.path.getsize(path) if os.path.exists(path) else None
    qp = QualityProfile(media_type="video", width=w, height=h, fps=float(fps),
                        duration_s=dur)
    if dur and size and w and h:
        qp.bitrate_kbps = size * 8 / dur / 1000.0
        qp.bits_per_pixel = size * 8 / (dur * fps * w * h)
        bpp = qp.bits_per_pixel
        qp.compression = "high" if bpp < 0.05 else "medium" if bpp < 0.12 else "low"
        if qp.compression == "high":
            qp.notes.append("Heavily compressed video: physiological and "
                            "sync experts are down-weighted.")
    if dur is not None and dur < 2:
        qp.notes.append("Very short clip (<2 s): temporal evidence is limited.")
    return qp


def audio_quality(waveform: np.ndarray, sr: int) -> QualityProfile:
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    qp = QualityProfile(media_type="audio", sample_rate=int(sr),
                        duration_s=float(len(x) / sr) if sr else None)
    if len(x) == 0:
        qp.notes.append("Empty audio.")
        return qp
    frame = max(int(0.02 * sr), 1)
    n = len(x) // frame
    if n >= 5:
        e = (x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12
        qp.snr_db = float(10 * np.log10(np.percentile(e, 90) / np.percentile(e, 10)))
    qp.clipping = bool(np.mean(np.abs(x) > 0.99) > 0.001)
    if sr and sr < 16000:
        qp.compression = "high"
        qp.notes.append(f"Low sample rate ({sr} Hz).")
    elif qp.snr_db is not None and qp.snr_db < 10:
        qp.compression = "medium"
        qp.notes.append("Noisy audio: acoustic detector is less reliable.")
    else:
        qp.compression = "low"
    return qp


def triage_image(source: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
    """Hash + quality + provenance for an image given as path/bytes/PIL."""
    raw: Optional[bytes] = None
    if isinstance(source, str):
        with open(source, "rb") as f:
            raw = f.read()
        img = Image.open(io.BytesIO(raw))
        img.load()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        img = Image.open(io.BytesIO(raw))
        img.load()
    else:
        img = source
    out: Dict[str, Any] = {
        "sha256": sha256_bytes(raw) if raw is not None else None,
        "quality": image_quality(img, raw),
        "provenance": detect_c2pa(raw) if raw is not None else {
            "c2pa_manifest_detected": False, "signature_verified": False,
            "note": "No file bytes available for provenance check."},
        "image": img,
    }
    return out
