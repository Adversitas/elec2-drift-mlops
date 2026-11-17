"""Command line: ``driftops benchmark | train | monitor | serve``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import data
from .monitoring import feature_drift
from .registry import Registry, load_model_dir, log_to_mlflow
from .simulate import DEFAULT_POLICIES, _acc, _fit, drift_profile, persistence_accuracy, run_policy


def _fmt_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cmd_benchmark(args: argparse.Namespace) -> int:
    df = data.load(args.data)
    batches = data.weekly_batches(df)
    if args.weeks:
        batches = batches[: args.weeks]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for kind in args.models:
        for policy in DEFAULT_POLICIES:
            summary, weekly = run_policy(batches, policy, kind=kind, warmup=args.warmup)
            summaries.append(summary)
            weekly.to_csv(out / f"weekly_{kind}_{policy.name}.csv", index=False)
            if args.mlflow:
                metrics = {k: v for k, v in summary.items() if isinstance(v, int | float)}
                log_to_mlflow(
                    "elec2-retraining-policies",
                    {
                        "policy": policy.name,
                        "model": kind,
                        "window": policy.window,
                        "gate": policy.gate,
                    },
                    metrics,
                )
            print(
                f"{kind:7s} {policy.name:22s} acc={summary['mean_accuracy']:.4f} "
                f"fits={summary['model_fits']:3d} {summary['wall_seconds']:.1f}s",
                flush=True,
            )
    table = pd.DataFrame(summaries)
    table.to_csv(out / "policies.csv", index=False)
    profile = drift_profile(batches, warmup=args.warmup)
    profile.to_csv(out / "drift_profile.csv", index=False)
    persist = persistence_accuracy(batches, args.warmup)
    shown = table[
        [
            "model",
            "policy",
            "mean_accuracy",
            "p10_week_accuracy",
            "worst_week_accuracy",
            "promotions",
            "model_fits",
            "wall_seconds",
        ]
    ]
    share_drift = (profile.drop(columns="week") > 0.25).mean().round(3).to_dict()
    md = [
        "# Retraining policies on Elec2",
        "",
        f"{len(batches)} weekly batches of {data.ROWS_PER_WEEK} half-hours; "
        f"{args.warmup} warm-up weeks; {len(batches) - args.warmup} weeks scored prequentially.",
        "",
        _fmt_table(shown),
        "",
        f"Persistence baseline (previous half-hour's label, needs real-time labels): {persist:.3f}",
        "",
        "Share of scored weeks where a feature's PSI vs the warm-up weeks exceeds 0.25: "
        + json.dumps(share_drift),
    ]
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {out / 'results.md'}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    df = data.load(args.data)
    batches = data.weekly_batches(df)
    end = args.end_week if args.end_week else len(batches)
    train = pd.concat(batches[max(0, end - args.window) : end - 1], ignore_index=True)
    valid = batches[end - 1]
    model = _fit(args.model, train)
    acc = _acc(model, valid)
    final = _fit(args.model, pd.concat([train, valid], ignore_index=True))
    reg = Registry(args.registry)
    ref = pd.concat([train, valid], ignore_index=True)[data.FEATURES]
    meta = {
        "model": args.model,
        "train_weeks": [end - args.window, end - 1],
        "holdout_accuracy": acc,
        "features": data.FEATURES,
    }
    version = reg.register(final, meta, reference=ref.sample(min(len(ref), 3000), random_state=0))
    champion = reg.champion_version()
    if champion is None or args.promote:
        reg.promote(version)
    print(
        json.dumps(
            {
                "version": version,
                "holdout_accuracy": round(acc, 4),
                "champion": reg.champion_version(),
            }
        )
    )
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    reg = Registry(args.registry)
    vdir = reg.path()
    reference = pd.read_csv(vdir / "reference.csv")
    rows = [json.loads(line)["input"] for line in Path(args.capture).read_text().splitlines()]
    current = pd.DataFrame(rows)
    drift = feature_drift(reference, current, data.MONITORED)
    alerts = {k: round(v, 3) for k, v in drift.items() if v > args.threshold}
    print(
        json.dumps(
            {
                "n": len(current),
                "psi": {k: round(v, 3) for k, v in drift.items()},
                "alerts": alerts,
            },
            indent=2,
        )
    )
    return 2 if alerts else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .serve import create_app

    if args.model_dir:
        model, meta = load_model_dir(args.model_dir)
    else:
        model, meta = Registry(args.registry).load()
    app = create_app(model, meta, capture_path=args.capture)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="driftops", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="replay Elec2 under every retraining policy")
    b.add_argument("--data", default=None)
    b.add_argument("--out", default="reports")
    b.add_argument("--warmup", type=int, default=4)
    b.add_argument("--weeks", type=int, default=0, help="limit the number of weeks (0 = all)")
    b.add_argument("--models", nargs="+", default=["hgb", "logreg"])
    b.add_argument("--mlflow", action="store_true", help="also log runs to MLflow")
    b.set_defaults(func=cmd_benchmark)

    t = sub.add_parser("train", help="train on a window, register, promote if no champion")
    t.add_argument("--data", default=None)
    t.add_argument("--registry", default="artifacts/registry")
    t.add_argument("--model", default="hgb", choices=["hgb", "logreg"])
    t.add_argument("--window", type=int, default=8)
    t.add_argument("--end-week", type=int, default=0)
    t.add_argument("--promote", action="store_true")
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("monitor", help="PSI of captured traffic vs the champion's reference")
    m.add_argument("--registry", default="artifacts/registry")
    m.add_argument("--capture", default="artifacts/capture.jsonl")
    m.add_argument("--threshold", type=float, default=0.25)
    m.set_defaults(func=cmd_monitor)

    s = sub.add_parser("serve", help="SageMaker-compatible server (/ping, /invocations)")
    s.add_argument("--registry", default="artifacts/registry")
    s.add_argument("--model-dir", default=None, help="e.g. /opt/ml/model on SageMaker")
    s.add_argument("--capture", default="artifacts/capture.jsonl")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
