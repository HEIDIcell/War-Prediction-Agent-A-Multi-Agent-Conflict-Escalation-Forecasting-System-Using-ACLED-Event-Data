"""Predict next-six-month conflict escalation risk for current cases.

This trains the model components on labelled historical cases and then applies
Baseline 1, Baseline 2, Single-Agent and Multi-Agent Debate to current cases.
"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.load_cases import load_market_cases, CONTEXT_FEATURE_COLUMNS
from src.models.traditional_models import make_logistic_regression, make_random_forest, fit_predict_model
from src.models.lstm_model import LSTMProbabilityModel
from src.agents.rag_retriever import CaseRetriever
from src.agents.single_agent import SingleRAGAgent
from src.agents.multi_agent_debate import MultiAgentDebateSystem


def risk_level(p: float) -> str:
    if p >= 0.80:
        return "Very high"
    if p >= 0.60:
        return "High"
    if p >= 0.30:
        return "Moderate"
    return "Low"


def load_current_cases(path="data/processed/current_cases.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("Current cases not found. Run: python -m src.data.build_current_cases")
    df = pd.read_csv(p)
    df["month_dt"] = pd.to_datetime(df["month"].astype(str) + "-01", errors="coerce")
    # Add optional feature columns that may be present in training data but not
    # in current cases, for example GDELT or UCDP auxiliary validation columns.
    for col in CONTEXT_FEATURE_COLUMNS:
        if col not in df.columns:
            if col == "cinc_ratio":
                df[col] = 1.0
            elif col == "trade_dependence":
                df[col] = 0.35
            else:
                df[col] = 0.0
    return df


def run():
    out_dir = Path("results/reports"); out_dir.mkdir(parents=True, exist_ok=True)
    train_df = load_market_cases("data/processed/market_cases_sample.csv")
    train_df = train_df[train_df["label"] >= 0].copy()
    current_df = load_current_cases("data/processed/current_cases.csv")
    # Ensure optional context columns exist by passing through load-case defaults would be cumbersome;
    # current builder already creates the full set.

    rows = []

    # Baseline 1: Logistic and Random Forest.
    for name, factory in [("Baseline1_LogisticRegression", make_logistic_regression), ("Baseline1_RandomForest", make_random_forest)]:
        start = time.perf_counter()
        probs = fit_predict_model(factory(), train_df, current_df)
        per_case_runtime = (time.perf_counter() - start) / max(len(current_df), 1)
        for (_, r), p in zip(current_df.iterrows(), probs):
            rows.append({
                "system": name, "case_id": r["case_id"], "dyad": r["dyad"], "input_month": r["month"],
                "horizon_months": 6, "predicted_probability": float(p), "risk_level": risk_level(float(p)),
                "runtime_seconds": per_case_runtime, "explanation": "Model-only probability from structured geo-context features.",
            })

    rf_probs = fit_predict_model(make_random_forest(), train_df, current_df)

    # Baseline 2: LSTM. If unavailable, use RF as a safe fallback and mark it clearly.
    try:
        start = time.perf_counter()
        lstm = LSTMProbabilityModel(sequence_length=6, hidden_size=24, epochs=25, lr=0.01)
        lstm_probs = lstm.fit_predict(train_df, current_df)
        per_case_runtime = (time.perf_counter() - start) / max(len(current_df), 1)
        sys_name = "Baseline2_LSTM"
    except Exception as exc:
        lstm_probs = rf_probs
        per_case_runtime = 0.0
        sys_name = "Baseline2_LSTM_FALLBACK_RF"
        print(f"[WARNING] LSTM current prediction failed; using RF fallback: {exc}")
    for (_, r), p in zip(current_df.iterrows(), lstm_probs):
        rows.append({
            "system": sys_name, "case_id": r["case_id"], "dyad": r["dyad"], "input_month": r["month"],
            "horizon_months": 6, "predicted_probability": float(p), "risk_level": risk_level(float(p)),
            "runtime_seconds": per_case_runtime, "explanation": "Sequence-model probability from the latest six-month feature window.",
        })

    retriever = CaseRetriever(k=3).fit(train_df)
    single = SingleRAGAgent(retriever)
    probs, explanations, runtimes = single.predict(current_df, rf_probs)
    for (_, r), p, e, rt in zip(current_df.iterrows(), probs, explanations, runtimes):
        rows.append({
            "system": "Single-Agent_RAG", "case_id": r["case_id"], "dyad": r["dyad"], "input_month": r["month"],
            "horizon_months": 6, "predicted_probability": float(p), "risk_level": risk_level(float(p)),
            "runtime_seconds": float(rt), "explanation": e,
        })

    multi = MultiAgentDebateSystem(retriever)
    probs, explanations, runtimes = multi.predict(current_df, rf_probs)
    for (_, r), p, e, rt in zip(current_df.iterrows(), probs, explanations, runtimes):
        rows.append({
            "system": "Multi-Agent_Debate", "case_id": r["case_id"], "dyad": r["dyad"], "input_month": r["month"],
            "horizon_months": 6, "predicted_probability": float(p), "risk_level": risk_level(float(p)),
            "runtime_seconds": float(rt), "explanation": e,
        })

    out = pd.DataFrame(rows)
    out_path = out_dir / "current_risk_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")
    print(out[["system", "dyad", "input_month", "predicted_probability", "risk_level"]])
    print("\n[NOTE] These are experimental risk estimates, not definitive claims that a state will start a war.")

if __name__ == "__main__":
    run()
