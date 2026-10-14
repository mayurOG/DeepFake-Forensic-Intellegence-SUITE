"""Explanations that are checked, not just drawn.

* ``occlusion_map`` - model-agnostic importance map: how much does the fake
  score drop when a region is hidden? Works with any backbone.
* ``deletion_faithfulness`` - tests whether that map is *faithful*: deleting
  the regions it ranks highest must lower the fake score more than deleting
  random regions. The score is shown next to the heatmap, so a decorative
  but wrong map is visibly flagged.
* ``build_narrative`` - a plain-language summary generated from measured
  fields only (no free-text model), so it cannot hallucinate artifacts.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

ScoreFn = Callable[[List[Image.Image]], np.ndarray]


def _mask_cell(arr: np.ndarray, i: int, j: int, grid: int, fill: np.ndarray):
    H, W = arr.shape[:2]
    y0, y1 = i * H // grid, (i + 1) * H // grid
    x0, x1 = j * W // grid, (j + 1) * W // grid
    arr[y0:y1, x0:x1] = fill


def occlusion_map(score_fn: ScoreFn, img: Image.Image, grid: int = 6) -> Dict:
    """Importance[i, j] = base_logit - logit_with_cell_hidden. Positive means
    the region supports a *fake* call."""
    base_img = img.convert("RGB")
    arr = np.asarray(base_img).copy()
    fill = arr.reshape(-1, 3).mean(axis=0).astype(arr.dtype)
    variants = []
    for i in range(grid):
        for j in range(grid):
            a = arr.copy()
            _mask_cell(a, i, j, grid, fill)
            variants.append(Image.fromarray(a))
    scores = np.asarray(score_fn([base_img] + variants), dtype=np.float64)
    base = float(scores[0])
    imp = (base - scores[1:]).reshape(grid, grid)
    return {"base_logit": base, "importance": imp, "grid": grid}


def overlay_heatmap(
    img: Image.Image, importance: np.ndarray, alpha: float = 0.55
) -> Image.Image:
    """Red overlay on regions that push the score towards 'fake'."""
    pos = np.clip(importance, 0, None)
    if pos.max() > 0:
        pos = pos / pos.max()
    heat = Image.fromarray((pos * 255).astype(np.uint8)).resize(
        img.size, Image.BICUBIC
    )
    h = np.asarray(heat, dtype=np.float32) / 255.0
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    out = base * (1 - alpha * h[..., None]) + red * (alpha * h[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def deletion_faithfulness(
    score_fn: ScoreFn, img: Image.Image, importance: np.ndarray,
    fractions: Sequence[float] = (0.1, 0.2, 0.35), n_random: int = 6,
    seed: int = 0,
) -> Dict:
    """Guided deletion vs random deletion.

    faithfulness in [-1, 1]: positive = hiding the map's top regions lowers
    the fake score more than hiding random regions (the map is informative);
    ~0 = no better than chance; negative = misleading. ``None`` when the
    image is not clearly scored as fake (nothing to explain).
    """
    grid = importance.shape[0]
    cells = grid * grid
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).copy()
    fill = arr.reshape(-1, 3).mean(axis=0).astype(arr.dtype)
    order = np.argsort(-importance.reshape(-1))
    rng = np.random.default_rng(seed)

    def delete(cell_ids):
        a = arr.copy()
        for c in cell_ids:
            _mask_cell(a, c // grid, c % grid, grid, fill)
        return Image.fromarray(a)

    guided, rand = [], []
    for f in fractions:
        k = max(1, int(round(f * cells)))
        guided.append(delete(order[:k]))
        for _ in range(n_random):
            rand.append(delete(rng.choice(cells, size=k, replace=False)))
    scores = np.asarray(score_fn([rgb] + guided + rand), dtype=np.float64)
    base = float(scores[0])
    if base < 0.5:
        return {"faithfulness": None, "reason": "image not scored as fake"}
    g = scores[1:1 + len(guided)]
    r = scores[1 + len(guided):].reshape(len(fractions), n_random).mean(axis=1)
    guided_drop = float(np.mean(base - g))
    random_drop = float(np.mean(base - r))
    f = (guided_drop - random_drop) / (abs(base) + 1e-6)
    return {
        "faithfulness": float(np.clip(f, -1.0, 1.0)),
        "guided_drop": guided_drop, "random_drop": random_drop,
        "base_logit": base,
    }


def _direction(logit: float) -> str:
    if logit > 0.5:
        return "points to manipulation"
    if logit < -0.5:
        return "points to authentic"
    return "is neutral"


def build_narrative(r: Dict) -> str:
    """Plain-language explanation built only from measured fields of an
    ``AnalysisResult.to_dict()``."""
    lines: List[str] = []
    verdict = r.get("verdict", "Inconclusive")
    p = r.get("p_fake")
    cal = r.get("calibrated", False)
    if not isinstance(p, (int, float)):
        pstr = "n/a"
    elif p > 0.99:
        pstr = ">99%"
    elif p < 0.01:
        pstr = "<1%"
    else:
        pstr = f"{p * 100:.0f}%"
    if verdict == "Inconclusive":
        lines.append(
            f"The system could not reach a decision it can stand behind "
            f"(estimated chance of manipulation: {pstr})."
        )
    else:
        lines.append(f"Verdict: {verdict} (estimated chance of manipulation: {pstr}).")
    if not cal:
        lines.append(
            "The detector has not been calibrated on your data, so these "
            "probabilities are rough guides, not guarantees."
        )
    if r.get("ood") is True:
        lines.append(
            "The media looks unlike the data the detector was calibrated on, "
            "so the result is treated with extra caution."
        )
    tier = r.get("tier")
    if tier == "tier1_early_exit":
        lines.append("The fast screen was confident, so no further experts were run.")
    used = [e for e in r.get("evidence", []) if e.get("status") == "used"]
    for e in sorted(used, key=lambda e: -abs(e.get("logit") or 0)):
        lines.append(
            f"- {e['name']}: {_direction(e.get('logit') or 0.0)} "
            f"(reliability {e.get('reliability', 0):.2f}"
            f"{', heuristic' if e.get('heuristic') else ''})."
        )
    skipped = [e for e in r.get("evidence", []) if e.get("status") == "skipped"]
    for e in skipped:
        lines.append(f"- {e['name']}: not used ({e.get('reason', 'unavailable')}).")
    st = r.get("stability")
    if st and st.get("stability_score") is not None:
        lines.append(
            f"Re-encoding stability: {st['stability_score'] * 100:.0f}% of "
            f"degraded copies kept the same call."
        )
    ex = r.get("explanation") or {}
    if ex.get("faithfulness") is not None:
        lines.append(
            f"Heatmap faithfulness check: {ex['faithfulness']:+.2f} "
            f"(positive = the highlighted regions really drive the score)."
        )
    for note in r.get("quality", {}).get("notes", []):
        lines.append(f"Note: {note}")
    return "\n".join(lines)
