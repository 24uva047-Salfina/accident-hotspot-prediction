"""Shared feature / label logic used by BOTH train_model.py and app.py.

Why this file exists
--------------------
In the old project the label (risk_level) was a direct function of
`total__2021`, and `total__2021` was also an input feature, so the model
only had to learn a threshold (100 % "accuracy", nothing learned).

Now the task is a genuine forecasting task:
    features  = what we knew in the PREVIOUS year (2020)
    target    = risk class of the NEXT year (2021)
At prediction time we feed the latest year (2021) to forecast the next one.
"""
import numpy as np
import pandas as pd

RISK_LABELS = ["Low", "Medium", "High"]          # 0, 1, 2
LOW_CUT, HIGH_CUT = 32.5, 65.0                    # on 0-100 min-max score

FEATURES = ["share_total", "share_fatal", "fatal_ratio", "latitude", "longitude"]


def risk_score(totals: pd.Series) -> pd.Series:
    """Min-max normalise yearly accident totals to a 0-100 score."""
    return (totals - totals.min()) / (totals.max() - totals.min()) * 100


def label_from_score(score: pd.Series) -> pd.Series:
    """0 = Low, 1 = Medium, 2 = High."""
    return pd.cut(score, bins=[-np.inf, LOW_CUT, HIGH_CUT, np.inf],
                  labels=[0, 1, 2]).astype(int)


def build_features(df: pd.DataFrame, fatal_col: str, total_col: str) -> pd.DataFrame:
    """Scale-free features: they do not change when the state total grows."""
    out = pd.DataFrame(index=df.index)
    out["share_total"] = df[total_col] / df[total_col].sum() * 100
    out["share_fatal"] = df[fatal_col] / df[fatal_col].sum() * 100
    out["fatal_ratio"] = df[fatal_col] / df[total_col]
    out["latitude"] = df["latitude"]
    out["longitude"] = df["longitude"]
    return out[FEATURES]
