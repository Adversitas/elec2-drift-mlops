"""Load the Elec2 stream and cut it into weekly batches.

Elec2 (Harries, 1999) records the New South Wales electricity market every 30 minutes from
May 1996 to December 1998. The target says whether the NSW price went UP or DOWN relative to
its moving average over the last 24 hours. Rows are in time order, 48 per day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PERIODS_PER_DAY = 48
DAYS_PER_WEEK = 7
ROWS_PER_WEEK = PERIODS_PER_DAY * DAYS_PER_WEEK

FEATURES = [
    "day",
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
    "period_sin",
    "period_cos",
]
MONITORED = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
TARGET = "up"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "electricity-normalized.arff"


def read_arff(path: str | Path) -> pd.DataFrame:
    """Minimal ARFF reader for the numeric and nominal attributes used by Elec2."""
    names: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if not in_data:
                if low.startswith("@attribute"):
                    names.append(line.split()[1].strip("'\""))
                elif low.startswith("@data"):
                    in_data = True
                continue
            rows.append([v.strip().strip("'\"") for v in line.split(",")])
    if not names or not rows:
        raise ValueError(f"no ARFF data found in {path}")
    return pd.DataFrame(rows, columns=names)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Type the columns, add cyclic time features, the target and the week index."""
    df = raw.copy()
    for col in ["date", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]:
        df[col] = df[col].astype(float)
    df["day"] = df["day"].astype(int)
    df[TARGET] = (df["class"] == "UP").astype(int)
    angle = 2 * np.pi * df["period"]
    df["period_sin"] = np.sin(angle)
    df["period_cos"] = np.cos(angle)
    df = df.reset_index(drop=True)
    df["week"] = df.index // ROWS_PER_WEEK
    # drop a trailing partial week so every batch has the same size
    counts = df["week"].value_counts()
    df = df[df["week"].map(counts) == ROWS_PER_WEEK].reset_index(drop=True)
    return df


def load(path: str | Path | None = None) -> pd.DataFrame:
    return prepare(read_arff(path or DEFAULT_PATH))


def weekly_batches(df: pd.DataFrame) -> list[pd.DataFrame]:
    return [g.reset_index(drop=True) for _, g in df.groupby("week", sort=True)]
