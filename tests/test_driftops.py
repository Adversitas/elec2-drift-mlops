import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from driftops import data
from driftops.cli import main
from driftops.monitoring import DriftMonitor, psi
from driftops.registry import Registry
from driftops.serve import create_app
from driftops.simulate import Policy, _fit, persistence_accuracy, run_policy


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return data.load()


@pytest.fixture(scope="module")
def batches(frame) -> list[pd.DataFrame]:
    return data.weekly_batches(frame)[:12]


def test_load_shapes(frame):
    assert len(frame) % data.ROWS_PER_WEEK == 0
    assert frame["week"].nunique() == len(frame) // data.ROWS_PER_WEEK
    assert set(frame[data.TARGET].unique()) == {0, 1}
    assert frame[data.FEATURES].notna().all().all()


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02
    assert psi(a, rng.normal(loc=2.0, size=5000)) > 1.0


def test_psi_constant_reference():
    ref = np.zeros(100)
    assert psi(ref, np.zeros(50)) == pytest.approx(0.0, abs=1e-3)
    assert psi(ref, np.ones(50)) > 1.0


def test_monitor_alerts_on_accuracy_drop(batches):
    mon = DriftMonitor(data.MONITORED, psi_threshold=float("inf"), acc_drop=0.05)
    mon.reset(batches[0], reference_acc=0.80)
    assert mon.check(batches[1], acc=0.78) == (False, [])
    alert, reasons = mon.check(batches[1], acc=0.70)
    assert alert and reasons[0].startswith("acc:")


def test_static_policy_never_retrains(batches):
    summary, weekly = run_policy(batches, Policy("static", "never"), kind="logreg", warmup=4)
    assert summary["weeks_scored"] == len(batches) - 4
    assert summary["model_fits"] == 2 and summary["promotions"] == 0
    assert not weekly["retrain"].any()


def test_periodic_policy_counts(batches):
    summary, _ = run_policy(batches, Policy("p", "periodic", every=4), kind="logreg", warmup=4)
    assert summary["promotions"] == (len(batches) - 4) // 4


def test_gate_fits_challenger_and_promotes_callback(batches):
    promoted = []
    policy = Policy("g", "always", gate=True)
    summary, weekly = run_policy(
        batches, policy, kind="logreg", warmup=4, on_promote=lambda m, info: promoted.append(info)
    )
    assert summary["model_fits"] == 2 + len(weekly) + summary["promotions"]
    assert len(promoted) == summary["promotions"]


def test_weekly_retraining_beats_static_on_logreg(batches):
    static, _ = run_policy(batches, Policy("s", "never"), kind="logreg")
    weekly, _ = run_policy(batches, Policy("w", "always"), kind="logreg")
    assert weekly["mean_accuracy"] > static["mean_accuracy"]


def test_persistence_baseline_in_range(batches):
    assert 0.5 < persistence_accuracy(batches, 4) < 1.0


def test_registry_roundtrip(tmp_path, batches):
    reg = Registry(tmp_path)
    model = _fit("logreg", batches[0])
    v1 = reg.register(model, {"note": "a"}, reference=batches[0][data.FEATURES])
    v2 = reg.register(model, {"note": "b"})
    assert (v1, v2) == (1, 2)
    with pytest.raises(LookupError):
        reg.load()
    reg.promote(v2)
    _, meta = reg.load()
    assert meta["note"] == "b" and reg.champion_version() == 2
    with pytest.raises(KeyError):
        reg.promote(7)


def test_serving_contract_and_capture(tmp_path, batches):
    model = _fit("logreg", batches[0])
    capture = tmp_path / "capture.jsonl"
    client = TestClient(create_app(model, {"version": 3}, capture_path=capture))
    assert client.get("/ping").json() == {"status": "ok", "model_version": 3}

    rows = batches[1][data.FEATURES].head(3)
    r = client.post("/invocations", json={"instances": rows.to_dict(orient="records")})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3 and all(p["label"] in {"UP", "DOWN"} for p in preds)

    csv = rows.to_csv(header=False, index=False)
    r = client.post("/invocations", content=csv, headers={"content-type": "text/csv"})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 3

    bad = client.post("/invocations", json={"instances": [{"day": 1}]})
    assert bad.status_code == 400
    lines = capture.read_text().splitlines()
    assert len(lines) == 6 and json.loads(lines[0])["model_version"] == 3


def test_cli_train_then_monitor(tmp_path, batches):
    reg = tmp_path / "reg"
    assert main(["train", "--registry", str(reg), "--model", "logreg", "--end-week", "10"]) == 0
    capture = tmp_path / "cap.jsonl"
    rows = batches[9][data.FEATURES].to_dict(orient="records")
    capture.write_text("\n".join(json.dumps({"input": r}) for r in rows))
    assert main(["monitor", "--registry", str(reg), "--capture", str(capture)]) in (0, 2)
