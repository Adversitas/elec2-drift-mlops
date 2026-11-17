"""Replay the Elec2 stream week by week under different retraining policies.

Protocol (prequential, no look-ahead):

1. The first ``warmup`` weeks train the first model. Its reference accuracy is measured on
   the last warm-up week with a model trained on the weeks before it.
2. For every following week the deployed model scores the whole week *before* its labels are
   known. Accuracy is recorded, then the labels are revealed (one-week label delay).
3. The policy decides whether to retrain. A challenger is trained on the window that ends just
   before the newest week and both models are scored on that newest week. With the gate on,
   the challenger replaces the champion only if it does at least as well. A promoted model is
   then refit on the window including the newest week.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, MONITORED, TARGET
from .monitoring import DriftMonitor, feature_drift


def make_model(kind: str = "hgb"):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    raise ValueError(f"unknown model kind {kind!r}")


@dataclass(frozen=True)
class Policy:
    name: str
    trigger: str  # "never" | "periodic" | "always" | "drift"
    every: int = 4
    window: int = 8  # weeks of history to train on; 0 = all history
    gate: bool = False
    psi_threshold: float = 0.25
    acc_drop: float = 0.05


INF = float("inf")

DEFAULT_POLICIES = [
    Policy("static", "never"),
    Policy("periodic-4w", "periodic", every=4),
    Policy("weekly", "always"),
    Policy("weekly-expanding", "always", window=0),
    Policy("psi-triggered", "drift", acc_drop=INF),
    Policy("perf-triggered-5pt", "drift", psi_threshold=INF, acc_drop=0.05),
    Policy("perf-triggered-2pt", "drift", psi_threshold=INF, acc_drop=0.02),
    Policy("perf-triggered-2pt+gate", "drift", psi_threshold=INF, acc_drop=0.02, gate=True),
]


def _window(history: list[pd.DataFrame], window: int, end: int | None = None) -> pd.DataFrame:
    hist = history[:end] if end is not None else history
    if window:
        hist = hist[-window:]
    return pd.concat(hist, ignore_index=True)


def _fit(kind: str, frame: pd.DataFrame):
    model = make_model(kind)
    model.fit(frame[FEATURES], frame[TARGET])
    return model


def _acc(model, frame: pd.DataFrame) -> float:
    return float((model.predict(frame[FEATURES]) == frame[TARGET].to_numpy()).mean())


def persistence_accuracy(batches: list[pd.DataFrame], warmup: int) -> float:
    """Naive baseline: predict the previous half-hour's label (needs real-time labels)."""
    y = pd.concat(batches, ignore_index=True)[TARGET].to_numpy()
    start = sum(len(b) for b in batches[:warmup])
    return float((y[start:] == y[start - 1 : -1]).mean())


def run_policy(
    batches: list[pd.DataFrame],
    policy: Policy,
    kind: str = "hgb",
    warmup: int = 4,
    on_promote=None,
) -> tuple[dict, pd.DataFrame]:
    if len(batches) <= warmup:
        raise ValueError("need more batches than warm-up weeks")
    history = list(batches[:warmup])
    t0 = time.perf_counter()
    probe = _fit(kind, _window(history, policy.window, end=-1))
    ref_acc = _acc(probe, history[-1])
    model = _fit(kind, _window(history, policy.window))
    fits, promotions = 2, 0
    monitor = DriftMonitor(MONITORED, policy.psi_threshold, policy.acc_drop)
    monitor.reset(_window(history, policy.window), ref_acc)

    rows = []
    for t in range(warmup, len(batches)):
        batch = batches[t]
        acc = _acc(model, batch)  # scored before labels are known
        history.append(batch)
        step = t - warmup + 1
        reasons: list[str] = []
        if policy.trigger == "never":
            retrain = False
        elif policy.trigger == "periodic":
            retrain = step % policy.every == 0
        elif policy.trigger == "always":
            retrain = True
        elif policy.trigger == "drift":
            retrain, reasons = monitor.check(batch, acc)
        else:
            raise ValueError(policy.trigger)

        promoted = False
        if retrain:
            promoted, new_ref = True, acc
            if policy.gate or policy.trigger == "drift":
                # out-of-sample estimate of the challenger on the newest labelled week; it is
                # the promotion test (with the gate) and the monitor's new reference accuracy
                challenger = _fit(kind, _window(history, policy.window, end=-1))
                fits += 1
                chall_acc = _acc(challenger, batch)
                if policy.gate:
                    promoted = chall_acc >= acc
                new_ref = chall_acc if promoted else acc
            if promoted:
                model = _fit(kind, _window(history, policy.window))
                fits += 1
                promotions += 1
                if on_promote is not None:
                    on_promote(model, {"week": t, "policy": policy.name, "reasons": reasons})
            monitor.reset(_window(history, policy.window), new_ref)
        rows.append(
            {
                "week": t,
                "accuracy": acc,
                "retrain": retrain,
                "promoted": promoted,
                "reasons": ";".join(reasons),
            }
        )
    elapsed = time.perf_counter() - t0
    weekly = pd.DataFrame(rows)
    summary = {
        "policy": policy.name,
        "model": kind,
        "weeks_scored": len(weekly),
        "mean_accuracy": float(weekly["accuracy"].mean()),
        "p10_week_accuracy": float(weekly["accuracy"].quantile(0.10)),
        "worst_week_accuracy": float(weekly["accuracy"].min()),
        "retrain_decisions": int(weekly["retrain"].sum()),
        "promotions": promotions,
        "model_fits": fits,
        "wall_seconds": elapsed,
    }
    return summary, weekly


def drift_profile(batches: list[pd.DataFrame], warmup: int = 4) -> pd.DataFrame:
    """PSI of each monitored feature, every week, against the warm-up weeks."""
    reference = pd.concat(batches[:warmup], ignore_index=True)
    rows = []
    for t in range(warmup, len(batches)):
        d = feature_drift(reference, batches[t], MONITORED)
        rows.append({"week": t, **d})
    return pd.DataFrame(rows)
