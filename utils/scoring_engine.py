from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No model artifact found. Checked:\n{checked}")


def _model_paths() -> dict[str, list[Path]]:
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
        ],
    }


def load_model():
    paths = _model_paths()
    model_path = _first_existing_path(paths["model"])
    scaler_path = _first_existing_path(paths["scaler"])
    features_path = _first_existing_path(paths["features"])

    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)
    with open(features_path, "rb") as handle:
        features = pickle.load(handle)

    return model, scaler, features


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Promise_To_Pay": "Promise_to_Pay",
        "Last_Payment_Days_Ago": "Last_Payment_Days",
        "Tenure_Months": "Tenure_Amount",
        "Customer Name": "Customer_Name",
        "Loan Tenure": "Tenure_Amount",
    }
    renamed = df.copy()
    for source, target in aliases.items():
        if source in renamed.columns and target not in renamed.columns:
            renamed = renamed.rename(columns={source: target})
    return renamed


def _get_bucket(pd_prob: float) -> str:
    if pd_prob >= 0.75:
        return "Critical"
    if pd_prob >= 0.50:
        return "High"
    if pd_prob >= 0.25:
        return "Medium"
    return "Low"


def _get_priority(pd_prob: float, dpd: float, emi: float) -> str:
    score = pd_prob * 50 + min(dpd / 180, 1) * 30 + min(emi / 45000, 1) * 20
    if score >= 60:
        return "P1 - Call immediately"
    if score >= 40:
        return "P2 - Call today"
    if score >= 20:
        return "P3 - Call this week"
    return "P4 - Low urgency"


def _get_action(bucket: str, promise_to_pay: int) -> str:
    actions = {
        "Critical": "Escalate to senior agent + field visit if PTP broken",
        "High": "Immediate call + send payment link",
        "Medium": "Scheduled callback + NACH re-presentation",
        "Low": "Automated IVR reminder",
    }
    action = actions.get(bucket, "Unknown")
    if promise_to_pay and bucket in ("High", "Critical"):
        action += " | PTP Follow-up"
    return action


def score_single(input_dict: dict, model, scaler, features) -> dict:
    row = _normalize_columns(pd.DataFrame([input_dict]))
    missing = [feature for feature in features if feature not in row.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = row[features]
    scaled = scaler.transform(row)
    pd_score = float(model.predict_proba(scaled)[0][1])
    bucket = _get_bucket(pd_score)
    priority = _get_priority(
        pd_score,
        float(row.iloc[0].get("DPD", 0)),
        float(row.iloc[0].get("EMI_Amount", 0)),
    )
    promise_to_pay = int(row.iloc[0].get("Promise_to_Pay", 0) or 0)

    return {
        "pd_score": round(pd_score * 100, 2),
        "risk_bucket": bucket,
        "priority": priority,
        "call_action": _get_action(bucket, promise_to_pay),
    }


def score_portfolio(df: pd.DataFrame, model, scaler, features) -> pd.DataFrame:
    df = _normalize_columns(df)
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded file: {missing}")

    x_scaled = scaler.transform(df[features])
    pd_scores = model.predict_proba(x_scaled)[:, 1]

    scored = df.copy()
    scored["PD_Score_%"] = np.round(pd_scores * 100, 2)
    scored["Risk_Bucket"] = scored["PD_Score_%"].apply(lambda score: _get_bucket(score / 100))
    scored["Priority_Rank"] = scored["PD_Score_%"].rank(ascending=False, method="first").astype(int)
    scored["Call_Action"] = scored.apply(
        lambda row: _get_action(
            _get_bucket(float(row["PD_Score_%"]) / 100),
            int(row.get("Promise_to_Pay", 0) or 0),
        ),
        axis=1,
    )
    return scored.sort_values("Priority_Rank")
