
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

def evaluate_predictions(y_true, y_prob, runtimes=None, explanation_scores=None, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    auc = np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_prob)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "brier_score": brier_score_loss(y_true, np.clip(y_prob, 0.001, 0.999)),
        "runtime_per_prediction": float(np.mean(runtimes)) if runtimes is not None else np.nan,
        "explanation_score": float(np.mean(explanation_scores)) if explanation_scores is not None else np.nan,
    }
    return out
