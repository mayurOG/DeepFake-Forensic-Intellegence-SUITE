"""The calibrated evidence cascade.

    triage -> Tier 1 fast screen -> [confident & in-domain? -> early exit]
           -> Tier 2 gated experts -> evidence fusion -> calibrate & abstain
           -> verdict set + explanation + stability + signed report

Public entry point: :class:`Pipeline` / :func:`get_pipeline`.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from detector import models as M
from detector.audio_io import has_audio_track, load_audio
from detector.calibration import (
    FAKE, REAL, CalibrationBundle, Calibrator, OODModel, sigmoid,
)
from detector.config import Settings, get_settings
from detector.experts.facesignals import (
    extract_face_signals, geometry_evidence, rppg_evidence, sync_evidence,
)
from detector.experts.tamper import tamper_evidence, tamper_heatmap_image
from detector.experts.temporal import (
    aggregate_frame_logits, load_temporal_head, temporal_evidence,
)
from detector.explain import (
    build_narrative, deletion_faithfulness, occlusion_map, overlay_heatmap,
)
from detector.fusion import Evidence, fuse, load_weights, skipped, _jsonable
from detector.stability import stability_report
from detector.triage import (
    audio_quality, image_quality, sha256_bytes, sha256_file, triage_image,
    video_quality, detect_c2pa, QualityProfile,
)

log = logging.getLogger("deepfake.pipeline")

IMAGE_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}
VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}
AUDIO_EXT = {"wav", "mp3", "flac", "ogg", "m4a"}
VALID_MODES = ("strict", "balanced", "sensitive")


def infer_media_type(name: str) -> Optional[str]:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    return None


@dataclass
class AnalysisResult:
    media_type: str
    filename: Optional[str] = None
    sha256: Optional[str] = None
    verdict: str = "Inconclusive"  # Real | Fake | Inconclusive
    label_set: List[str] = field(default_factory=list)
    p_fake: float = 0.5
    confidence: Optional[float] = None
    calibrated: bool = False
    calibrator: Dict[str, Any] = field(default_factory=dict)
    ood: Optional[bool] = None
    ood_ratio: Optional[float] = None
    tier: str = "tier2"
    mode: str = "balanced"
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    fusion: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    stability: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    backends: List[str] = field(default_factory=list)
    narrative: str = ""
    # Not serialised: PIL images etc. for the UI, and raw scores for fitting.
    artifacts: Dict[str, Any] = field(default_factory=dict, repr=False)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("artifacts", "raw")}
        return _jsonable(d)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def aggregate_faces(logits: np.ndarray) -> float:
    """One image-level logit from several face logits. One manipulated face
    among many should flag the image, but a single noisy face should not
    dominate, so blend the max with the mean."""
    z = np.asarray(logits, dtype=np.float64)
    if len(z) == 0:
        return 0.0
    return float(0.5 * z.max() + 0.5 * z.mean())


def sample_frames(path: str, n: int) -> List[Tuple[int, float, Image.Image]]:
    """``n`` evenly spaced frames as (index, seconds, PIL image)."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out: List[Tuple[int, float, Image.Image]] = []
    if total > 0:
        idxs = np.unique(np.linspace(0, total - 1, n + 2)[1:-1].astype(int))
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok and fr is not None:
                out.append((int(i), float(i / fps),
                            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    if len(out) < max(2, n // 2):  # seeking unreliable: sequential fallback
        cap.release()
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        out = []
        if frames:
            idxs = np.unique(np.linspace(0, len(frames) - 1, n).astype(int))
            for i in idxs:
                out.append((int(i), float(i / fps),
                            Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))))
    cap.release()
    return out


def frames_at(path: str, times: Sequence[float]) -> List[Tuple[int, float, Image.Image]]:
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    out = []
    for t in times:
        idx = max(0, int(round(t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok and fr is not None:
            out.append((idx, float(idx / fps),
                        Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    cap.release()
    return out


def _largest_face_crop(img: Image.Image, detector) -> Tuple[Image.Image, Optional[tuple], int]:
    faces = detector.detect(img)
    if not faces:
        return img, None, 0
    f = max(faces, key=lambda ff: ff.area)
    return M.crop_with_margin(img, f.box), f.box, len(faces)


class Pipeline:
    def __init__(
        self, settings: Optional[Settings] = None,
        calibration: Optional[CalibrationBundle] = None,
        ood: Optional[Dict[str, OODModel]] = None,
        fusion_weights: Optional[Dict[str, float]] = None,
        temporal_head=None,
    ):
        self.settings = settings or get_settings()
        self.calibration = calibration
        self.ood = ood
        self.fusion_weights = fusion_weights
        self.temporal_head = temporal_head
        self._explicit = (calibration is not None, ood is not None,
                          fusion_weights is not None, temporal_head is not None)
        self.reload()

    # -- artifacts -------------------------------------------------------
    def reload(self) -> None:
        d = self.settings.artifact_dir
        if not self._explicit[0]:
            self.calibration = CalibrationBundle.load(os.path.join(d, "calibration.json"))
        if not self._explicit[1]:
            self.ood = {m: OODModel.load(os.path.join(d, f"ood_{m}.npz"))
                        for m in ("image", "video", "audio")}
        if not self._explicit[2]:
            self.fusion_weights = load_weights(d)
        if not self._explicit[3]:
            try:
                self.temporal_head = load_temporal_head(os.path.join(d, "temporal_head.pt"))
            except Exception as e:  # pragma: no cover
                log.warning("Could not load temporal head: %s", e)
                self.temporal_head = None

    def status(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"device": M.get_device(), "calibrators": {}}
        for name, cal in (self.calibration.items or {}).items():
            out["calibrators"][name] = {"fitted": cal.fitted, "n_cal": cal.n_cal,
                                        "alpha": cal.alpha}
        out["any_calibrated"] = any(c.fitted for c in self.calibration.items.values())
        out["ood_fitted"] = {k: v.fitted for k, v in self.ood.items()}
        out["learned_fusion_weights"] = self.fusion_weights is not None
        out["learned_temporal_head"] = self.temporal_head is not None
        out["image_backends"] = [b.name for b in M.get_image_backends()]
        out["audio_backend"] = M.get_audio_backend().name
        return out

    # -- decision ----------------------------------------------------------
    def _decide(self, cal_name: str, logit: float, ood_name: str,
                emb: Optional[np.ndarray], mode: str):
        cal = self.calibration.get(cal_name)
        p = float(cal.prob_fake(logit))
        label_set = cal.label_set(p, mode)
        ood_flag, ratio = (None, None)
        if emb is not None:
            ood_flag, ratio = self.ood.get(ood_name, OODModel()).is_ood(emb[None] if emb.ndim == 1 else emb)
        verdict = (FAKE if label_set == [FAKE] else REAL if label_set == [REAL]
                   else "Inconclusive")
        reason = None
        if ood_flag and verdict != "Inconclusive":
            verdict, reason = "Inconclusive", "outside_calibrated_domain"
        return cal, p, label_set, verdict, ood_flag, ratio, reason

    def _finalise(self, res: AnalysisResult, cal: Calibrator, p, label_set,
                  verdict, ood_flag, ratio, reason, mode, tier):
        res.p_fake = float(p)
        res.label_set = list(label_set)
        res.verdict = verdict
        res.confidence = (p if verdict == FAKE else 1.0 - p if verdict == REAL else None)
        res.calibrated = bool(cal.fitted)
        res.calibrator = {"fitted": cal.fitted, "alpha": cal.alpha, "n_cal": cal.n_cal}
        res.ood, res.ood_ratio = ood_flag, ratio
        res.tier, res.mode = tier, mode
        if reason == "outside_calibrated_domain":
            res.warnings.append(
                "Input looks unlike the calibration data; the verdict was "
                "downgraded to Inconclusive.")
        if not cal.fitted:
            res.warnings.append(
                "Detector not calibrated: probabilities are uncalibrated and "
                "decisions use a conservative fallback margin. Run "
                "`python -m train.fit_calibration` on labelled data.")

    # -- entry point -------------------------------------------------------
    def analyze(
        self, source, media_type: Optional[str] = None,
        filename: Optional[str] = None, mode: str = "balanced",
        explain: bool = True, stability: bool = True,
        force_full: bool = False, audio_chunk_s: int = 3,
    ) -> AnalysisResult:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        tmp_path = None
        try:
            if hasattr(source, "read"):  # Streamlit UploadedFile / file object
                filename = filename or getattr(source, "name", None)
                source = source.read()
            if media_type is None:
                if isinstance(source, Image.Image):
                    media_type = "image"
                else:
                    name = filename or (source if isinstance(source, str) else "")
                    media_type = infer_media_type(str(name))
            if media_type is None:
                raise ValueError("Cannot infer media type; pass media_type=")
            if media_type != "image" and isinstance(source, (bytes, bytearray)):
                suffix = "." + (filename.rsplit(".", 1)[-1] if filename and "." in filename else "bin")
                fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, "wb") as f:
                    f.write(source)
                source = tmp_path
            if isinstance(source, str) and not filename:
                filename = os.path.basename(source)
            if media_type == "image":
                return self._analyze_image(source, filename, mode, explain, stability, force_full)
            if media_type == "video":
                return self._analyze_video(source, filename, mode, explain, stability, force_full)
            if media_type == "audio":
                return self._analyze_audio(source, filename, mode, explain, audio_chunk_s)
            raise ValueError(f"Unsupported media type {media_type}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ======================================================================
    # IMAGE
    # ======================================================================
    def _analyze_image(self, source, filename, mode, explain, do_stability, force_full):
        t_all = time.time()
        res = AnalysisResult(media_type="image", filename=filename)
        tri = triage_image(source)
        img: Image.Image = tri["image"].convert("RGB")
        quality: QualityProfile = tri["quality"]
        res.sha256, res.quality, res.provenance = tri["sha256"], quality.to_dict(), tri["provenance"]
        t = time.time()

        detector = M.get_face_detector()
        faces = detector.detect(img)
        faces = sorted(faces, key=lambda f: -f.area)[:6]
        if faces:
            crops = [M.crop_with_margin(img, f.box) for f in faces]
            boxes = [f.box for f in faces]
        else:
            crops, boxes = [img], []
            res.warnings.append(
                "No face detected: the whole image was analysed (detectors "
                "trained on faces are less reliable here).")
        res.timings["faces"] = time.time() - t

        backends = M.get_image_backends()
        res.backends = [b.name for b in backends]
        t = time.time()
        z0, emb0 = backends[0].predict(crops)
        res.timings["tier1"] = time.time() - t
        z_img = aggregate_faces(z0)
        ev_neural = Evidence(
            "neural", z_img, quality.neural_reliability(),
            detail={"backend": backends[0].name,
                    "per_face_logits": [round(float(v), 3) for v in z0],
                    "faces": len(faces)})
        mean_emb = None if emb0 is None else np.asarray(emb0).mean(axis=0)
        res.raw.update({"tier1_logit": z_img, "embedding": mean_emb})

        # cheap provenance / metadata evidence
        meta, ev_meta = self._metadata_evidence(source, tri)
        res.metadata = meta

        evidence: List[Evidence] = [ev_neural]
        if ev_meta is not None:
            evidence.append(ev_meta)

        cal1, p1, set1, v1, ood1, r1, reason1 = self._decide(
            "image.tier1", z_img, "image", mean_emb, mode)
        confident = (v1 in (FAKE, REAL)) and (ood1 is not True)
        if ev_meta is not None and v1 == REAL:
            confident = False  # an explicit AI-generator tag contradicts "Real"
        early = confident and not force_full

        face_box = boxes[0] if boxes else None
        t = time.time()
        if early:
            res.tier = "tier1_early_exit"
            cal, p, label_set, verdict, ood_flag, ratio, reason = cal1, p1, set1, v1, ood1, r1, reason1
            final_logit = z_img
            res.fusion = {"logit": z_img, "disagreement": 0.0, "n_used": len(evidence)}
            # list the experts that were deliberately not run
            for name in ("tamper", "forensic_prior"):
                evidence.append(skipped(name, "not needed: fast screen was confident", True))
        else:
            res.tier = "tier2"
            # additional backbones (heterogeneous ensemble)
            for b in backends[1:]:
                zb, _ = b.predict(crops)
                evidence.append(Evidence(
                    "neural_probe", aggregate_faces(zb), quality.neural_reliability(),
                    detail={"backend": b.name}))
            evidence.append(tamper_evidence(img, face_box, quality.compression))
            evidence.append(self._forensic_prior(img))
            fr = fuse(evidence, self.fusion_weights)
            final_logit = fr.logit
            res.fusion = {"logit": fr.logit, "disagreement": fr.disagreement,
                          "n_used": fr.n_used, "contributions": fr.contributions}
            cal, p, label_set, verdict, ood_flag, ratio, reason = self._decide(
                "image.final", final_logit, "image", mean_emb, mode)
        res.timings["tier2" if not early else "decide"] = time.time() - t
        res.raw["final_logit"] = final_logit
        res.evidence = [e.to_dict() for e in evidence]
        self._finalise(res, cal, p, label_set, verdict, ood_flag, ratio, reason, mode, res.tier)

        # explanation + stability (UI features; do not affect the verdict)
        primary = crops[0]
        score_fn = lambda ims: backends[0].predict(ims)[0]
        res.artifacts["annotated"] = self._annotate(img, boxes)
        res.artifacts["face_crops"] = [c.resize((224, 224)) for c in crops]
        if explain:
            t = time.time()
            om = occlusion_map(score_fn, primary, grid=6)
            ff = deletion_faithfulness(score_fn, primary, om["importance"])
            res.artifacts["heatmap"] = overlay_heatmap(primary, om["importance"])
            res.explanation = {"method": "occlusion", "grid": 6,
                               "faithfulness": ff.get("faithfulness"),
                               "faithfulness_detail": ff,
                               "importance": om["importance"].round(4).tolist()}
            from utils.visualizers import generate_ela_image, generate_fft_spectrum
            res.artifacts["ela"] = generate_ela_image(img)
            res.artifacts["fft"] = generate_fft_spectrum(img)
            if any(e.name == "tamper" and e.status == "used" for e in evidence) or not early:
                from detector.experts.tamper import tamper_map
                tm = tamper_map(img, face_box)
                res.artifacts["tamper_map"] = tamper_heatmap_image(tm, img.size)
            res.timings["explain"] = time.time() - t
        if do_stability:
            t = time.time()

            def p_of(ims):
                z, _ = backends[0].predict(ims)
                return float(cal1.prob_fake(aggregate_faces(z)))

            res.stability = stability_report(crops, p_of, base_p=float(p1))
            res.timings["stability"] = time.time() - t
        res.narrative = build_narrative(res.to_dict())
        res.timings["total"] = time.time() - t_all
        return res

    def _metadata_evidence(self, source, tri) -> Tuple[Optional[Dict], Optional[Evidence]]:
        from detector.metadata_detector import analyze_metadata

        meta = None
        try:
            if isinstance(source, str):
                meta = analyze_metadata(source)
            elif isinstance(source, (bytes, bytearray)):
                meta = analyze_metadata(io.BytesIO(bytes(source)))
        except Exception:
            meta = None
        if meta and meta.get("ai_generator_signature"):
            sig = meta["ai_generator_signature"]
            return meta, Evidence(
                "metadata_ai", 6.0, 1.0,
                detail={"signature": sig},
                reason="File metadata names an AI generator.")
        return meta, None

    def _forensic_prior(self, img: Image.Image) -> Evidence:
        from detector.facial_forensics import analyze_image_quality_and_noise

        q = analyze_image_quality_and_noise(img)
        risk = float(q["heuristic_risk_score"])
        return Evidence("forensic_prior", (risk - 0.25) * 2.0, 0.5, heuristic=True,
                        detail={k: q[k] for k in
                                ("laplacian_blur_score", "chroma_variance",
                                 "srm_noise_energy", "heuristic_risk_score")})

    @staticmethod
    def _annotate(img: Image.Image, boxes) -> Image.Image:
        out = img.copy()
        d = ImageDraw.Draw(out)
        for b in boxes:
            d.rectangle(list(b), outline="#ef4444", width=3)
        return out

    # ======================================================================
    # VIDEO
    # ======================================================================
    def _frame_scores(self, frames, backend):
        """Per-frame fake logits on the largest face crop (full frame if no
        face). Returns logits, embeddings, crops, boxes."""
        detector = M.get_face_detector()
        crops, boxes, no_face = [], [], 0
        for _, _, im in frames:
            crop, box, n = _largest_face_crop(im, detector)
            crops.append(crop)
            boxes.append(box)
            no_face += int(box is None)
        z, emb = backend.predict(crops)
        return np.asarray(z, dtype=np.float64), emb, crops, boxes, no_face

    def _analyze_video(self, path, filename, mode, explain, do_stability, force_full):
        t_all = time.time()
        res = AnalysisResult(media_type="video", filename=filename)
        res.sha256 = sha256_file(path)
        quality = video_quality(path)
        res.quality = quality.to_dict()
        res.provenance = detect_c2pa(open(path, "rb").read(2_000_000))
        backends = M.get_image_backends()
        res.backends = [b.name for b in backends]
        s = self.settings

        t = time.time()
        frames = sample_frames(path, s.tier1_frames)
        if not frames:
            raise ValueError("Could not decode any frame from the video.")
        z, emb, crops, boxes, no_face = self._frame_scores(frames, backends[0])
        times = [f[1] for f in frames]
        res.timings["tier1_visual"] = time.time() - t
        if no_face == len(frames):
            res.warnings.append("No faces found in sampled frames; whole frames were analysed.")
        elif no_face:
            res.warnings.append(f"No face in {no_face} of {len(frames)} sampled frames.")

        qf = quality.neural_reliability()
        ev_temporal = temporal_evidence(z, times, emb, self.temporal_head, qf)

        # audio (cheap relative to the benefits: catches voice-cloned audio)
        ev_audio, audio_wave, audio_info = self._video_audio_evidence(path)
        evidence: List[Evidence] = [ev_temporal, ev_audio]

        fr1 = fuse([e for e in evidence], self.fusion_weights)
        mean_emb = None if emb is None else np.asarray(emb).mean(axis=0)
        cal1, p1, set1, v1, ood1, r1, reason1 = self._decide(
            "video.tier1", fr1.logit, "video", mean_emb, mode)
        res.raw.update({"tier1_logit": fr1.logit, "embedding": mean_emb})
        confident = (v1 in (FAKE, REAL)) and (ood1 is not True)
        early = confident and not force_full

        t = time.time()
        if early:
            res.tier = "tier1_early_exit"
            cal, p, label_set, verdict, ood_flag, ratio, reason = cal1, p1, set1, v1, ood1, r1, reason1
            final_logit = fr1.logit
            res.fusion = {"logit": fr1.logit, "disagreement": fr1.disagreement,
                          "n_used": fr1.n_used, "contributions": fr1.contributions}
            for name in ("sync", "rppg", "geometry"):
                evidence.append(skipped(name, "not needed: fast screen was confident", True))
        else:
            res.tier = "tier2"
            # 1) densify around the most suspicious frames
            if len(z) >= 3:
                top = np.argsort(-z)[:2]
                extra_t = []
                for i in top:
                    for dt in (-0.6, -0.3, 0.3, 0.6):
                        tt = times[i] + dt
                        if tt >= 0:
                            extra_t.append(tt)
                extra = frames_at(path, extra_t)
                if extra:
                    z2, emb2, crops2, boxes2, nf2 = self._frame_scores(extra, backends[0])
                    allt = times + [f[1] for f in extra]
                    allz = np.concatenate([z, z2])
                    alle = None if emb is None or emb2 is None else np.concatenate([emb, emb2])
                    order = np.argsort(allt)
                    allt = [allt[i] for i in order]
                    allz = allz[order]
                    alle = None if alle is None else alle[order]
                    evidence[0] = temporal_evidence(allz, allt, alle, self.temporal_head, qf)
                    evidence[0].detail["densified_frames"] = len(extra)
                    z, times = allz, allt
            # 2) face-signal experts on a short consecutive window
            sig = None
            try:
                sig = extract_face_signals(
                    path, M.get_face_detector(), max_seconds=s.face_signal_seconds)
            except Exception as e:
                log.warning("face signal extraction failed: %s", e)
            evidence.append(rppg_evidence(sig, quality.compression))
            evidence.append(geometry_evidence(sig))
            evidence.append(sync_evidence(sig, audio_wave, 16000, quality.compression))
            # 3) extra backbones on the sampled crops
            for b in backends[1:]:
                zb, _ = b.predict(crops)
                evidence.append(Evidence("neural_probe", aggregate_frame_logits(zb),
                                         qf, detail={"backend": b.name}))
            fr = fuse(evidence, self.fusion_weights)
            final_logit = fr.logit
            res.fusion = {"logit": fr.logit, "disagreement": fr.disagreement,
                          "n_used": fr.n_used, "contributions": fr.contributions}
            cal, p, label_set, verdict, ood_flag, ratio, reason = self._decide(
                "video.final", final_logit, "video", mean_emb, mode)
        res.timings["tier2" if not early else "decide"] = time.time() - t
        res.raw["final_logit"] = final_logit
        res.evidence = [e.to_dict() for e in evidence]
        self._finalise(res, cal, p, label_set, verdict, ood_flag, ratio, reason, mode, res.tier)

        # modality disagreement note
        if ev_audio.status == "used" and ev_temporal.logit is not None:
            if (ev_audio.logit > 1.0 and ev_temporal.logit < -1.0):
                res.warnings.append("Audio looks synthetic while the visuals look authentic.")
            elif (ev_audio.logit < -1.0 and ev_temporal.logit > 1.0):
                res.warnings.append("Visuals look manipulated while the audio looks authentic.")

        # timeline for the UI (calibrated per-frame probability)
        cal_t = self.calibration.get("video.tier1")
        for tt, zz in zip(times, z):
            res.timeline.append({"timestamp": round(float(tt), 3),
                                 "fake_probability": float(cal_t.prob_fake(zz)),
                                 "logit": float(zz)})
        res.segments = [{"start": a, "end": b} for a, b in
                        ev_temporal.detail.get("suspicious_segments", [])]
        # artifacts / explanation
        k = int(np.argmax(z[:len(crops)])) if len(crops) else 0
        res.artifacts["frames"] = [(f[1], f[2]) for f in frames]
        res.artifacts["suspicious_frames"] = [
            (frames[i][1], frames[i][2]) for i in np.argsort(-z[:len(frames)])[:4]]
        if explain and crops:
            score_fn = lambda ims: backends[0].predict(ims)[0]
            om = occlusion_map(score_fn, crops[k], grid=6)
            ff = deletion_faithfulness(score_fn, crops[k], om["importance"])
            res.artifacts["heatmap"] = overlay_heatmap(crops[k], om["importance"])
            res.explanation = {"method": "occlusion", "grid": 6,
                               "frame_time": round(float(frames[k][1]), 2),
                               "faithfulness": ff.get("faithfulness"),
                               "faithfulness_detail": ff}
        if do_stability and crops:
            top_k = list(np.argsort(-z[:len(crops)])[:3])
            sel = [crops[i] for i in top_k]

            def p_of(ims):
                zz, _ = backends[0].predict(ims)
                return float(cal1.prob_fake(aggregate_frame_logits(zz)))

            res.stability = stability_report(sel, p_of)
        if audio_info:
            res.segments += audio_info.get("segments", [])
        res.narrative = build_narrative(res.to_dict())
        res.timings["total"] = time.time() - t_all
        return res

    def _video_audio_evidence(self, path):
        try:
            if not has_audio_track(path):
                return skipped("audio", "no audio track"), None, None
            wave = load_audio(path, 16000)
            if wave is None or len(wave) < 16000:
                return skipped("audio", "audio shorter than 1 s"), wave, None
            ev, info = self._audio_evidence(wave, 3)
            return ev, wave, info
        except Exception as e:
            log.warning("audio analysis failed: %s", e)
            return skipped("audio", f"audio analysis failed: {e}"), None, None

    # ======================================================================
    # AUDIO
    # ======================================================================
    @staticmethod
    def _chunks(wave: np.ndarray, sr: int, chunk_s: int):
        n = int(chunk_s * sr)
        chunks, spans = [], []
        for i in range(0, len(wave), n):
            c = wave[i:i + n]
            if len(c) < n:
                if len(chunks) and len(c) < sr:  # drop a sub-1s tail
                    break
                c = np.concatenate([c, np.zeros(n - len(c), dtype=c.dtype)])
            chunks.append(c)
            spans.append((i / sr, min((i + n) / sr, len(wave) / sr)))
        return chunks, spans

    def _audio_evidence(self, wave: np.ndarray, chunk_s: int):
        sr = 16000
        backend = M.get_audio_backend()
        chunks, spans = self._chunks(wave, sr, chunk_s)
        z, emb = backend.predict(chunks, sr)
        aq = audio_quality(wave, sr)
        rel = aq.neural_reliability()
        ev = Evidence("audio", aggregate_frame_logits(z), rel,
                      detail={"backend": backend.name, "chunks": len(chunks),
                              "max_chunk_logit": float(np.max(z))})
        segs = [{"start": float(a), "end": float(b), "logit": float(zz)}
                for (a, b), zz in zip(spans, z)]
        return ev, {"segments": segs, "logits": z, "emb": emb, "spans": spans,
                    "quality": aq, "chunks": chunks}

    def _analyze_audio(self, path, filename, mode, explain, chunk_s):
        t_all = time.time()
        res = AnalysisResult(media_type="audio", filename=filename)
        res.sha256 = sha256_file(path)
        wave = load_audio(path, 16000)
        if wave is None or len(wave) < 8000:
            raise ValueError("Could not decode audio, or it is shorter than 0.5 s.")
        res.backends = [M.get_audio_backend().name]
        ev, info = self._audio_evidence(wave, chunk_s)
        res.quality = info["quality"].to_dict()
        evidence = [ev]
        fr = fuse(evidence, self.fusion_weights)
        emb = info["emb"]
        mean_emb = None if emb is None else np.asarray(emb).mean(axis=0)
        cal, p, label_set, verdict, ood_flag, ratio, reason = self._decide(
            "audio.final", fr.logit, "audio", mean_emb, mode)
        res.tier = "single_stage"
        res.fusion = {"logit": fr.logit, "disagreement": fr.disagreement,
                      "n_used": fr.n_used, "contributions": fr.contributions}
        res.raw.update({"tier1_logit": fr.logit, "final_logit": fr.logit,
                        "embedding": mean_emb})
        res.evidence = [e.to_dict() for e in evidence]
        self._finalise(res, cal, p, label_set, verdict, ood_flag, ratio, reason, mode, res.tier)
        for seg, zz in zip(info["segments"], info["logits"]):
            pp = float(cal.prob_fake(zz))
            res.timeline.append({"timestamp": round((seg["start"] + seg["end"]) / 2, 3),
                                 "fake_probability": pp,
                                 "label": "Fake" if pp >= 0.5 else "Real"})
        res.segments = [{"start": s["start"], "end": s["end"],
                         "fake_probability": float(cal.prob_fake(s["logit"]))}
                        for s in info["segments"]]
        res.artifacts["waveform"] = wave
        if explain:
            from utils.visualizers import generate_mel_spectrogram
            res.artifacts["mel"] = generate_mel_spectrogram(wave, 16000)
        res.narrative = build_narrative(res.to_dict())
        res.timings["total"] = time.time() - t_all
        return res


_PIPELINE: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE


def reset_pipeline() -> None:
    global _PIPELINE
    _PIPELINE = None
