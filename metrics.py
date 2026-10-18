"""metrics.py -- binary screening metrics + patient-cluster bootstrap (no torch)."""
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, roc_auc_score, roc_curve)

import config


def safe_auroc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_auprc(y, p) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def youden_threshold(y, p) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[i], 0.0, 1.0))


def sens_at_spec(y, p, spec: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = np.where(1 - fpr >= spec)[0]
    return float(tpr[ok].max()) if len(ok) else 0.0


def spec_at_sens(y, p, sens: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = np.where(tpr >= sens)[0]
    return float((1 - fpr[ok]).max()) if len(ok) else 0.0


def confusion_at(y, p, thr: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan")}


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def summarize(y, p, thr: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "prevalence": float(y.mean()) if len(y) else float("nan"),
           "auroc": safe_auroc(y, p), "auprc": safe_auprc(y, p), "threshold": float(thr)}
    if len(np.unique(y)) == 2:
        for s in config.SPECIFICITY_TARGETS:
            out[f"sens@spec{int(s*100)}"] = sens_at_spec(y, p, s)
        for s in config.SENSITIVITY_TARGETS:
            out[f"spec@sens{int(s*100)}"] = spec_at_sens(y, p, s)
        out["brier"] = float(brier_score_loss(y, p))
        out["ece"] = expected_calibration_error(y, p)
    pred = (p >= thr).astype(int)
    out["balanced_acc"] = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out.update(confusion_at(y, p, thr))
    return out


def cluster_bootstrap_ci(y, p, groups: Sequence, fn: Callable, iters: int = config.BOOTSTRAP_ITERS,
                         seed: int = 0, alpha: float = 0.05) -> Optional[tuple]:
    """Resample PATIENTS (not images) with replacement -> honest CIs when a
    patient contributes several images."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        v = fn(y[ii], p[ii])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 50:
        return None
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
