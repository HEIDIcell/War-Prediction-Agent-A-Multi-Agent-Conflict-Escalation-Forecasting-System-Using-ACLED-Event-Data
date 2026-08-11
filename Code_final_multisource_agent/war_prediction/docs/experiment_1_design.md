# Experiment 1: Architecture Comparison

## Research question
Does a multi-agent debate architecture improve predictive performance and robustness compared with model-only, deep-learning-only, and single-agent baselines?

## Systems
- Baseline 1: Logistic Regression / Random Forest
- Baseline 2: LSTM only
- Single-Agent: local RAG-style risk analyst
- Multi-Agent Debate: Sentiment, Escalation, De-escalation and Data-driven agents with a Judge Agent

## Metrics
Accuracy, Precision, Recall, F1-score, ROC-AUC, Brier Score, Explanation Score, Runtime.

## Explanation Score
The project uses an automated reproducible score instead of human scoring. It rewards:
- references to conflict/cooperation/sentiment/model evidence
- counter-evidence
- probability consistency
- use of similar retrieved historical cases
