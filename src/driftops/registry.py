"""A small file-based model registry with versions and a champion alias.

Layout: ``<root>/<name>/v<N>/model.joblib`` + ``meta.json``, and ``<root>/<name>/champion``
(a text file holding the champion version). Runs can also be mirrored to MLflow.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class Registry:
    def __init__(self, root: str | Path, name: str = "elec2-updown") -> None:
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name

    def versions(self) -> list[int]:
        return sorted(int(p.name[1:]) for p in self.dir.glob("v*") if p.name[1:].isdigit())

    def register(self, model: Any, meta: dict, reference: pd.DataFrame | None = None) -> int:
        """Store a model, its metadata and (optionally) a sample of its training features,
        which the drift monitor later uses as the reference distribution."""
        version = (self.versions() or [0])[-1] + 1
        vdir = self.dir / f"v{version}"
        vdir.mkdir()
        joblib.dump(model, vdir / "model.joblib")
        meta = {"version": version, "registered_at": time.time(), **meta}
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
        if reference is not None:
            reference.to_csv(vdir / "reference.csv", index=False)
        return version

    def path(self, version: int | None = None) -> Path:
        version = version or self.champion_version()
        if version is None:
            raise LookupError("registry has no champion")
        return self.dir / f"v{version}"

    def promote(self, version: int) -> None:
        if version not in self.versions():
            raise KeyError(f"unknown version {version}")
        (self.dir / "champion").write_text(str(version))

    def champion_version(self) -> int | None:
        f = self.dir / "champion"
        return int(f.read_text()) if f.exists() else None

    def load(self, version: int | None = None) -> tuple[Any, dict]:
        return load_model_dir(self.path(version))


def load_model_dir(vdir: str | Path) -> tuple[Any, dict]:
    """Load ``model.joblib`` + ``meta.json`` from one directory (e.g. SageMaker's
    ``/opt/ml/model``)."""
    vdir = Path(vdir)
    return joblib.load(vdir / "model.joblib"), json.loads((vdir / "meta.json").read_text())


def log_to_mlflow(experiment: str, params: dict, metrics: dict) -> bool:
    """Mirror a run to MLflow when it is installed. Returns False when it is not."""
    try:
        import mlflow
    except ImportError:
        return False
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=params.get("policy")):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    return True
