"""Noise robustness experiment for the War Prediction Agent.

This experiment tests how sensitive the best agent-based system is to noisy
geopolitical observations. It perturbs numeric input features in the test set
and reruns the Multi-Agent Debate system across multiple random seeds.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_cases import load_market_cases, FEATURE_COLUMNS
from src.experiments.run_architecture_comparison import temporal_train_test_split
from src.models.traditional_models import make_random_forest
from src.data.load_cases import FEATURE_COLUMNS
from src.agents.rag_retriever import CaseRetriever
from src.agents.multi_agent_debate import MultiAgentDebateSystem
from src.analysis.metrics import evaluate_predictions


def perturb_features(test_df: pd.DataFrame, train_df: pd.DataFrame, noise_level: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = test_df.copy()
    if noise_level <= 0:
        return out
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            continue
        scale = float(pd.to_numeric(train_df[col], errors="coerce").std())
        if not np.isfinite(scale) or scale == 0:
            scale = max(abs(float(pd.to_numeric(train_df[col], errors="coerce").mean())), 1.0)
        noise = rng.normal(0, noise_level * scale, size=len(out))
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).values + noise
        # Counts/probabilities should remain in plausible ranges.
        if col.endswith("count") or col in ["fatalities_sum", "us_actor_events", "recent_conflicts"]:
            out[col] = np.clip(out[col], 0, None)
        if col == "model_probability":
            out[col] = np.clip(out[col], 0.001, 0.999)
    return out


def run(noise_levels=(0.0, 0.1, 0.2, 0.3, 0.4), seeds=range(10)):
    out_tables = Path("results/tables"); out_tables.mkdir(parents=True, exist_ok=True)
    out_figs = Path("results/figures"); out_figs.mkdir(parents=True, exist_ok=True)
    df = load_market_cases("data/processed/market_cases_sample.csv")
    train_df, test_df = temporal_train_test_split(df)
    y_test = test_df["label"].values

    retriever = CaseRetriever(k=3).fit(train_df)
    rf_model = make_random_forest()
    rf_model.fit(train_df[FEATURE_COLUMNS], train_df["label"].values)
    rows = []
    for nl in noise_levels:
        for seed in seeds:
            noisy_test = perturb_features(test_df, train_df, nl, seed)
            rf_probs = np.clip(rf_model.predict_proba(noisy_test[FEATURE_COLUMNS])[:, 1], 0.001, 0.999)
            agent = MultiAgentDebateSystem(retriever)
            probs, explanations, runtimes = agent.predict(noisy_test, rf_probs)
            metrics = evaluate_predictions(y_test, probs, runtimes=runtimes)
            metrics.update({"noise_level": nl, "seed": seed, "system": "MultiAgent_Debate"})
            rows.append(metrics)
    raw = pd.DataFrame(rows)
    raw.to_csv(out_tables / "noise_robustness_raw.csv", index=False)

    agg = raw.groupby(["system", "noise_level"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        brier_score_mean=("brier_score", "mean"), brier_score_std=("brier_score", "std"),
        runtime_mean=("runtime_per_prediction", "mean"), runtime_std=("runtime_per_prediction", "std"),
    ).reset_index()
    agg.to_csv(out_tables / "noise_robustness_summary.csv", index=False)

    # Separate figures, as advised in the coursework slides.
    plt.figure(figsize=(7, 4))
    plt.errorbar(agg["noise_level"], agg["brier_score_mean"], yerr=agg["brier_score_std"], marker="o", capsize=4)
    plt.xlabel("Feature noise level")
    plt.ylabel("Mean Brier Score")
    plt.title("Noise Robustness: Probability Error")
    plt.tight_layout()
    plt.savefig(out_figs / "noise_robustness_brier.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.errorbar(agg["noise_level"], agg["roc_auc_mean"], yerr=agg["roc_auc_std"], marker="o", capsize=4)
    plt.xlabel("Feature noise level")
    plt.ylabel("Mean ROC-AUC")
    plt.title("Noise Robustness: Ranking Performance")
    plt.tight_layout()
    plt.savefig(out_figs / "noise_robustness_roc_auc.png", dpi=180)
    plt.close()

    print("[INFO] Wrote noise robustness outputs to results/tables and results/figures")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    run()
