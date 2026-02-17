from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"txn_id", "date", "account", "vendor", "amount", "description"}


def validate_ledger(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Ledger file missing columns: {sorted(missing)}")


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    validate_ledger(df)
    data = df.copy()
    data["amount"] = data["amount"].astype(float)
    data["abs_amount"] = data["amount"].abs()

    mean = data["abs_amount"].mean()
    std = data["abs_amount"].std(ddof=0)
    std = std if std > 0 else 1.0
    data["z_score"] = (data["abs_amount"] - mean) / std

    keyword_mask = data["description"].str.contains(
        r"urgent|manual|adjustment|override", case=False, regex=True
    )
    score = np.maximum(data["z_score"], 0) + keyword_mask.astype(int) * 0.8
    data["score"] = score.round(3)

    anomaly_mask = (data["z_score"] >= z_threshold) | (keyword_mask & (data["z_score"] > 1.2))
    anomalies = data.loc[anomaly_mask, ["txn_id", "account", "amount", "description", "z_score", "score"]].copy()
    anomalies["reason"] = anomalies.apply(
        lambda row: "High amount outlier" if row["z_score"] >= z_threshold else "Risky keyword + elevated amount",
        axis=1,
    )

    return anomalies.sort_values("score", ascending=False).reset_index(drop=True)
