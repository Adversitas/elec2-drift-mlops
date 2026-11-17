"""SageMaker-compatible inference server.

SageMaker starts a custom container with ``docker run <image> serve`` and talks to it on port
8080 through two routes: ``GET /ping`` (health) and ``POST /invocations`` (inference). This app
implements that contract and appends every request to a JSONL data-capture file, the same idea
as SageMaker data capture, so the drift monitor can read production inputs later.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response

from .data import FEATURES


def _to_frame(body: bytes, content_type: str) -> pd.DataFrame:
    if content_type.startswith("text/csv"):
        frame = pd.read_csv(io.BytesIO(body), header=None)
        if frame.shape[1] != len(FEATURES):
            raise ValueError(f"CSV rows need {len(FEATURES)} columns in order: {FEATURES}")
        frame.columns = FEATURES
        return frame
    if content_type.startswith("application/json"):
        payload = json.loads(body)
        instances = payload.get("instances") if isinstance(payload, dict) else payload
        if not isinstance(instances, list) or not instances:
            raise ValueError('JSON body must be {"instances": [{feature: value, ...}, ...]}')
        frame = pd.DataFrame(instances)
        missing = [f for f in FEATURES if f not in frame.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")
        return frame[FEATURES]
    raise ValueError(f"unsupported content type {content_type!r}")


def create_app(model, meta: dict, capture_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="driftops inference", version=str(meta.get("version", "?")))
    capture = Path(capture_path) if capture_path else None
    if capture:
        capture.parent.mkdir(parents=True, exist_ok=True)

    @app.get("/ping")
    def ping() -> dict:
        return {"status": "ok", "model_version": meta.get("version")}

    @app.post("/invocations")
    async def invocations(request: Request) -> Response:
        start = time.perf_counter()
        content_type = request.headers.get("content-type", "application/json")
        try:
            frame = _to_frame(await request.body(), content_type)
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        proba = model.predict_proba(frame[FEATURES])[:, 1]
        preds = [{"label": "UP" if p >= 0.5 else "DOWN", "p_up": round(float(p), 4)} for p in proba]
        latency_ms = (time.perf_counter() - start) * 1000
        if capture:
            with capture.open("a", encoding="utf-8") as fh:
                for row, pred in zip(frame.to_dict(orient="records"), preds, strict=True):
                    record = {"ts": time.time(), "model_version": meta.get("version")}
                    fh.write(json.dumps({**record, "input": row, "output": pred}) + "\n")
        body = json.dumps({"predictions": preds, "latency_ms": round(latency_ms, 2)})
        return Response(content=body, media_type="application/json")

    return app
