"""Drift statistics: population stability index, KS test and a retraining monitor."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with quantile bins taken from the reference sample.

    Rule of thumb: below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 major shift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        # constant reference: compare "same value" vs "different value" shares
        ref_pct = np.array([1.0 - eps, eps])
        same = float(np.isclose(current, reference[0]).mean())
        cur_pct = np.clip(np.array([same, 1.0 - same]), eps, None)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_pvalue(reference: np.ndarray, current: np.ndarray) -> float:
    return float(stats.ks_2samp(reference, current).pvalue)


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> dict:
    return {f: psi(reference[f].to_numpy(), current[f].to_numpy()) for f in features}


@dataclass
class DriftMonitor:
    """Decides when to retrain, from feature drift and from delayed-label accuracy.

    * ``psi_threshold``: alert if any monitored feature's PSI exceeds it.
    * ``acc_drop``: alert if accuracy on the latest labelled batch falls more than this
      below the accuracy measured when the current model was deployed.
    """

    features: list[str]
    psi_threshold: float = 0.25
    acc_drop: float = 0.05
    reference: pd.DataFrame | None = None
    reference_acc: float | None = None
    history: list[dict] = field(default_factory=list)

    def reset(self, reference: pd.DataFrame, reference_acc: float) -> None:
        self.reference = reference
        self.reference_acc = reference_acc

    def check(self, batch: pd.DataFrame, acc: float) -> tuple[bool, list[str]]:
        if self.reference is None or self.reference_acc is None:
            raise RuntimeError("monitor has no reference; call reset() first")
        drift = feature_drift(self.reference, batch, self.features)
        reasons = [f"psi:{f}={v:.2f}" for f, v in drift.items() if v > self.psi_threshold]
        if acc < self.reference_acc - self.acc_drop:
            reasons.append(f"acc:{acc:.3f}<{self.reference_acc:.3f}-{self.acc_drop}")
        self.history.append({"max_psi": max(drift.values()), "acc": acc, "alert": bool(reasons)})
        return bool(reasons), reasons
