
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_cases import load_market_cases
from src.models.traditional_models import make_logistic_regression, make_random_forest, fit_predict_model
from src.models.lstm_model import LSTMProbabilityModel
from src.agents.rag_retriever import CaseRetriever
from src.agents.single_agent import SingleRAGAgent
from src.agents.multi_agent_debate import MultiAgentDebateSystem
from src.analysis.metrics import evaluate_predictions
from src.analysis.explanation_score import explanation_score
from src.analysis.plots import plot_architecture_metrics

def temporal_train_test_split(df, test_size=0.30):
    unique_months = sorted(df["month_dt"].dropna().unique())
    if len(unique_months) < 4:
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df["label"])
    cutoff_idx = int(len(unique_months) * (1 - test_size))
    cutoff_month = unique_months[cutoff_idx]
    train_df = df[df["month_dt"] < cutoff_month].copy()
    test_df = df[df["month_dt"] >= cutoff_month].copy()
    if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["label"] if df["label"].nunique() > 1 else None)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def run():
    out_tables = Path("results/tables"); out_tables.mkdir(parents=True, exist_ok=True)
    df = load_market_cases("data/processed/market_cases_sample.csv")
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows")
    print("[INFO] Label distribution:")
    print(df["label"].value_counts())

    train_df, test_df = temporal_train_test_split(df)
    y_test = test_df["label"].values
    results, prediction_rows = [], []

    for name, model_factory in [
        ("Baseline1_LogisticRegression", make_logistic_regression),
        ("Baseline1_RandomForest", make_random_forest),
    ]:
        start = time.perf_counter()
        probs = fit_predict_model(model_factory(), train_df, test_df)
        total_runtime = time.perf_counter() - start
        runtimes = np.full(len(test_df), total_runtime / max(len(test_df), 1))
        metrics = evaluate_predictions(y_test, probs, runtimes=runtimes)
        metrics["system"] = name
        results.append(metrics)
        for case_id, p in zip(test_df["case_id"], probs):
            prediction_rows.append({"system": name, "case_id": case_id, "probability": p, "explanation": ""})

    rf_probs = fit_predict_model(make_random_forest(), train_df, test_df)

    try:
        start = time.perf_counter()
        lstm = LSTMProbabilityModel(sequence_length=6, hidden_size=12, epochs=3, lr=0.01)
        lstm_probs = lstm.fit_predict(train_df, test_df)
        total_runtime = time.perf_counter() - start
        runtimes = np.full(len(test_df), total_runtime / max(len(test_df), 1))
        metrics = evaluate_predictions(y_test, lstm_probs, runtimes=runtimes)
        metrics["system"] = "Baseline2_LSTM"
        results.append(metrics)
        for case_id, p in zip(test_df["case_id"], lstm_probs):
            prediction_rows.append({"system": "Baseline2_LSTM", "case_id": case_id, "probability": p, "explanation": ""})
    except Exception as exc:
        print(f"[WARNING] LSTM failed or PyTorch unavailable: {exc}")
        probs = np.full(len(test_df), train_df["label"].mean())
        metrics = evaluate_predictions(y_test, probs, runtimes=np.zeros(len(test_df)))
        metrics["system"] = "Baseline2_LSTM_FAILED_FALLBACK"
        results.append(metrics)

    retriever = CaseRetriever(k=3).fit(train_df)

    single_agent = SingleRAGAgent(retriever)
    probs, explanations, runtimes = single_agent.predict(test_df, rf_probs)
    exp_scores = [explanation_score(e, p) for e, p in zip(explanations, probs)]
    metrics = evaluate_predictions(y_test, probs, runtimes=runtimes, explanation_scores=exp_scores)
    metrics["system"] = "SingleAgent_RAG"
    results.append(metrics)
    for case_id, p, e in zip(test_df["case_id"], probs, explanations):
        prediction_rows.append({"system": "SingleAgent_RAG", "case_id": case_id, "probability": p, "explanation": e})

    multi_agent = MultiAgentDebateSystem(retriever)
    probs, explanations, runtimes = multi_agent.predict(test_df, rf_probs)
    exp_scores = [explanation_score(e, p) for e, p in zip(explanations, probs)]
    metrics = evaluate_predictions(y_test, probs, runtimes=runtimes, explanation_scores=exp_scores)
    metrics["system"] = "MultiAgent_Debate"
    results.append(metrics)
    for case_id, p, e in zip(test_df["case_id"], probs, explanations):
        prediction_rows.append({"system": "MultiAgent_Debate", "case_id": case_id, "probability": p, "explanation": e})

    results_df = pd.DataFrame(results)
    cols = ["system", "accuracy", "precision", "recall", "f1", "roc_auc", "brier_score", "explanation_score", "runtime_per_prediction"]
    results_df = results_df[cols]
    results_df.to_csv(out_tables / "architecture_comparison_full.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out_tables / "predictions_by_system.csv", index=False)
    plot_architecture_metrics(results_df)

    print("\n[INFO] Architecture comparison results:")
    print(results_df.to_string(index=False))
    print("\n[INFO] Outputs written to results/tables and results/figures.")

if __name__ == "__main__":
    run()
