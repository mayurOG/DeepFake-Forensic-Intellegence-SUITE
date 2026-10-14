"""Evaluation metrics: AUC, EER, TPR at fixed FPR, calibration and selective
(abstain-aware) metrics. Pure numpy/sklearn so it runs anywhere."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = _arr(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, _arr(score)))


def eer(y_true: Sequence[int], score: Sequence[float]) -> float:
    """Equal error rate (lower is better)."""
    y = _arr(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, _arr(score))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def tpr_at_fpr(
    y_true: Sequence[int], score: Sequence[float], target_fpr: float = 0.01
) -> float:
    """Fraction of fakes caught while keeping false alarms on real media
    at or below ``target_fpr``."""
    y = _arr(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, _arr(score))
    ok = np.where(fpr <= target_fpr)[0]
    return float(tpr[ok].max()) if len(ok) else 0.0


def expected_calibration_error(
    y_true: Sequence[int], p_fake: Sequence[float], bins: int = 15
) -> float:
    y = _arr(y_true)
    p = _arr(p_fake)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def brier(y_true: Sequence[int], p_fake: Sequence[float]) -> float:
    return float(np.mean((_arr(p_fake) - _arr(y_true)) ** 2))


def selective_metrics(
    y_true: Sequence[int], verdicts: Sequence[str]
) -> Dict[str, Optional[float]]:
    """Metrics for a system that may answer 'Inconclusive'.

    * abstain_rate        - share of inputs left undecided
    * accuracy_answered   - accuracy on the inputs it did decide
    * false_alarm_rate    - real media called Fake (among answered reals)
    * miss_rate           - fakes called Real (among answered fakes)
    """
    y = np.asarray(y_true).astype(int).reshape(-1)
    v = np.asarray(list(verdicts)).reshape(-1)
    answered = v != "Inconclusive"
    out: Dict[str, Optional[float]] = {
        "n": int(len(y)),
        "abstain_rate": float(1.0 - answered.mean()) if len(y) else None,
    }
    if answered.any():
        pred = (v[answered] == "Fake").astype(int)
        out["accuracy_answered"] = float((pred == y[answered]).mean())
    else:
        out["accuracy_answered"] = None
    real_ans = answered & (y == 0)
    fake_ans = answered & (y == 1)
    out["false_alarm_rate"] = (
        float((v[real_ans] == "Fake").mean()) if real_ans.any() else None
    )
    out["miss_rate"] = (
        float((v[fake_ans] == "Real").mean()) if fake_ans.any() else None
    )
    return out
