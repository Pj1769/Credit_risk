"""
utils/scoring_engine.py
Smart Collections Prioritization Agent — Bajaj Finance DMS
Loads the trained logistic regression PD model and scores accounts.
Auto-retrains if pickle files are missing or incompatible.
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS = [
    "DPD", "Bounce_Count", "Contact_Attempts",
    "Successful_Contacts", "Promise_to_Pay",
    "Last_Payment_Days", "Credit_Score",
    "EMI_Amount", "Loan_Amount", "Tenure_Amount",
]

TARGET_COL = "Default_Label"


# ── Internal helpers ───────────────────────────────────────────────────────────
def _candidate_paths() -> dict:
    return {
        "model": [
            PROJECT_ROOT / "models" / "pd_model.pkl",
            PROJECT_ROOT / "pd_model.pkl",
        ],
        "scaler": [
            PROJECT_ROOT / "models" / "scaler.pkl",
            PROJECT_ROOT / "feature_scaler.pkl",
            PROJECT_ROOT / "scaler.pkl",
        ],
        "features": [
            PROJECT_ROOT / "models" / "feature_names.pkl",
            PROJECT_ROOT / "feature_cols.pkl",
            PROJECT_ROOT / "feature_names.pkl",
        ],
    }


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# ── Auto trainer ───────────────────────────────────────────────────────────────
def _train_fresh():
    """Train a new model from loan_accounts.xlsx and save fresh pkl files."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    data_path = PROJECT_ROOT / "data" / "loan_accounts.xlsx"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Make sure data/loan_accounts.xlsx is in your repository."
        )

    df = pd.read_excel(data_path)

    # Normalise column name aliases
    rename_map = {
        "Tenure_Months":         "Tenure_Amount",
        "Promise_To_Pay":        "Promise_to_Pay",
        "Last_Payment_Days_Ago": "Last_Payment_Days",
    }
    for src, tgt in rename_map.items():
        if src in df.columns and tgt not in df.columns:
            df = df.rename(columns={src: tgt})

    # Build target column if absent
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = (df["DPD"] >= 90).astype(int)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=42, solver="lbfgs"
    )
    model.fit(X_train_s, y_train)

    # Save to models/ folder
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    with open(models_dir / "pd_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(models_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    return model, scaler, FEATURE_COLS


# ── Public loader ──────────────────────────────────────────────────────────────
def load_model():
    """
    Load model artifacts from disk.
    If files are missing OR incompatible (pickle version mismatch),
    automatically retrains from the raw dataset.
    """
    candidates = _candidate_paths()

    model_path   = _first_existing(candidates["model"])
    scaler_path  = _first_existing(candidates["scaler"])
    features_path = _first_existing(candidates["features"])

    if model_path and scaler_path and features_path:
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            with open(features_path, "rb") as f:
                features = pickle.load(f)
            return model, scaler, features
        except Exception:
            # Pickle incompatibility — retrain silently
            pass

    # Either files missing or failed to load — retrain
    return _train_fresh()


# ── Risk classification ────────────────────────────────────────────────────────
def _classify(pd_score: float):
    if pd_score >= 75:
        bucket   = "Critical"
        priority = "Priority 1 — Call immediately"
        action   = ("⚠️ Immediate outbound call required. Escalate to senior collections agent. "
                    "Discuss settlement options, NACH re-presentation, and legal notice risk.")
    elif pd_score >= 50:
        bucket   = "High"
        priority = "Priority 2 — Call today"
        action   = ("📞 Schedule call within today's shift. Focus on PTP collection and NACH "
                    "mandate reactivation. Offer EMI restructuring if needed.")
    elif pd_score >= 25:
        bucket   = "Medium"
        priority = "Priority 3 — Call this week"
        action   = ("🔔 Queue for this week's outbound campaign. Send WhatsApp/SMS reminder "
                    "first. Collect PTP date and monitor.")
    else:
        bucket   = "Low"
        priority = "Priority 4 — Monitor"
        action   = ("✅ Low default risk. Include in automated IVR/SMS campaign. "
                    "No manual call needed unless DPD increases.")
    return bucket, priority, action


# ── Single account scorer ──────────────────────────────────────────────────────
def score_single(input_data: dict, model, scaler, features: list) -> dict:
    row = pd.DataFrame([{f: input_data.get(f, 0) for f in features}])

    if scaler is not None:
        row_scaled = scaler.transform(row)
    else:
        row_scaled = row.values

    pd_prob  = model.predict_proba(row_scaled)[0][1]
    pd_score = round(pd_prob * 100, 1)
    bucket, priority, action = _classify(pd_score)

    return {
        "pd_score":    pd_score,
        "risk_bucket": bucket,
        "priority":    priority,
        "call_action": action,
    }


# ── Portfolio scorer ───────────────────────────────────────────────────────────
def score_portfolio(df: pd.DataFrame, model, scaler, features: list) -> pd.DataFrame:
    # Normalise column aliases
    rename_map = {
        "Tenure_Months":         "Tenure_Amount",
        "Promise_To_Pay":        "Promise_to_Pay",
        "Last_Payment_Days_Ago": "Last_Payment_Days",
    }
    for src, tgt in rename_map.items():
        if src in df.columns and tgt not in df.columns:
            df = df.rename(columns={src: tgt})

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded file: {missing}")

    X = df[features].fillna(0)
    X_scaled = scaler.transform(X) if scaler is not None else X.values

    pd_probs = model.predict_proba(X_scaled)[:, 1]
    df = df.copy()
    df["PD_Score_%"] = (pd_probs * 100).round(1)

    buckets, priorities, actions = [], [], []
    for score in df["PD_Score_%"]:
        b, p, a = _classify(score)
        buckets.append(b)
        priorities.append(p)
        actions.append(a)

    df["Risk_Bucket"] = buckets
    df["Priority"]    = priorities
    df["Call_Action"] = actions
    df = df.sort_values("PD_Score_%", ascending=False).reset_index(drop=True)
    df.insert(0, "Priority_Rank", range(1, len(df) + 1))

    return df
