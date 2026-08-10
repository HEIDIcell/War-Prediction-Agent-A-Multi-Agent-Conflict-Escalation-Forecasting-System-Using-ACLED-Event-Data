"""Evaluate individual heads inside the Multi-Agent Debate system.

This supports discussion of which specialised reasoning head contributes most:
Sentiment, Escalation, De-escalation, Geo-context, Data-driven, and the full Judge.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.load_cases import load_market_cases
from src.experiments.run_architecture_comparison import temporal_train_test_split
from src.models.traditional_models import make_random_forest, fit_predict_model
from src.agents.rag_retriever import CaseRetriever
from src.agents.multi_agent_debate import MultiAgentDebateSystem
from src.analysis.metrics import evaluate_predictions


def run():
    out_dir = Path("results/tables"); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_market_cases("data/processed/market_cases_sample.csv")
    df = df[df["label"] >= 0].copy()
    train_df, test_df = temporal_train_test_split(df)
    y = test_df["label"].values
    rf_probs = fit_predict_model(make_random_forest(), train_df, test_df)
    retriever = CaseRetriever(k=3).fit(train_df)
    system = MultiAgentDebateSystem(retriever)

    head_probs = {
        "SentimentAgent": [],
        "EscalationAgent": [],
        "DeescalationAgent": [],
        "GeoContextAgent": [],
        "DataDrivenAgent": [],
        "Full_MultiAgent_Judge": [],
    }
    full_probs, _, _ = system.predict(test_df, rf_probs)
    for (_, row), mp, fp in zip(test_df.iterrows(), rf_probs, full_probs):
        head_probs["SentimentAgent"].append(system.sentiment.assess(row)[0])
        head_probs["EscalationAgent"].append(system.escalation.assess(row)[0])
        head_probs["DeescalationAgent"].append(system.deescalation.assess(row)[0])
        head_probs["GeoContextAgent"].append(system.geo_context.assess(row)[0])
        head_probs["DataDrivenAgent"].append(system.data.assess(row, mp)[0])
        head_probs["Full_MultiAgent_Judge"].append(fp)

    rows = []
    for head, probs in head_probs.items():
        metrics = evaluate_predictions(y, np.asarray(probs), runtimes=np.zeros(len(y)))
        metrics["agent_head"] = head
        rows.append(metrics)
    out = pd.DataFrame(rows)
    out = out[["agent_head"] + [c for c in out.columns if c != "agent_head"]]
    out_path = out_dir / "multi_head_agent_evaluation.csv"
    out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}")
    print(out)

if __name__ == "__main__":
    run()
