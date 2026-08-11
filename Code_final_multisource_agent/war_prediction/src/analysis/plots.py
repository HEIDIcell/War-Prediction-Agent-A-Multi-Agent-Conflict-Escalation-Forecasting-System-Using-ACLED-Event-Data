
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_architecture_metrics(results_df: pd.DataFrame, output_dir="results/figures"):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    display_names = {
        "Baseline1_LogisticRegression": "Logistic Regression",
        "Baseline1_RandomForest": "Random Forest",
        "Baseline2_LSTM": "LSTM only",
        "SingleAgent_RAG": "Single-Agent RAG",
        "MultiAgent_Debate": "Multi-Agent Debate",
    }
    df = results_df.copy()
    df["display_name"] = df["system"].map(display_names).fillna(df["system"])
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "explanation_score"]
    ax = df.set_index("display_name")[metrics].plot(kind="bar", figsize=(12, 6))
    ax.set_title("Architecture Comparison across Evaluation Metrics")
    ax.set_ylabel("Score"); ax.set_xlabel("System"); ax.set_ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(output_dir / "architecture_metrics_bar.png", dpi=220); plt.close()

    for metric, title, filename, ylabel in [
        ("brier_score", "Brier Score Comparison", "brier_score_comparison.png", "Brier Score (lower is better)"),
        ("roc_auc", "ROC-AUC Comparison", "roc_auc_comparison.png", "ROC-AUC (higher is better)"),
        ("runtime_per_prediction", "Runtime Comparison", "runtime_comparison.png", "Seconds per prediction"),
        ("explanation_score", "Explanation Score Comparison", "explanation_score_comparison.png", "Explanation Score"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df["display_name"], df[metric])
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("System")
        plt.xticks(rotation=25, ha="right"); plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=220); plt.close()
