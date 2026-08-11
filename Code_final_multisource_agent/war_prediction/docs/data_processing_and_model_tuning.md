# Data Processing and Model Tuning Guide

## Data choice

The final implementation uses the registered ACLED event-level CSV as the main recent dataset. This avoids the year mismatch between recent ACLED data and older COW MID labels.

- Main event data: ACLED 2016-2025 event-level rows.
- Main label: future six-month high-escalation label derived from ACLED event intensity.
- Optional context: World Bank or fallback geo-context features.
- Not used as recent training label: COW MID, because it does not cover 2016-2025.

## Handling year mismatch

COW MID and COW NMC are useful background datasets, but they are not temporally aligned with the 2016-2025 ACLED file. Therefore, the project keeps the supervised task internally consistent by deriving both features and future labels from ACLED. World Bank indicators are optional annual context features and can be disabled with `--no-worldbank`.

## Feature construction

For each US-target dyad-month, the system aggregates:

- `conflict_count`
- `military_count`
- `threat_count`
- `cooperation_count`
- `diplomatic_count`
- `avg_tone`
- `avg_goldstein`
- `fatalities_sum`
- `us_actor_events`
- `cinc_ratio`
- `diplomatic_score`
- `trade_dependence`
- `news_sentiment`
- `recent_conflicts`

ACLED is not a sentiment dataset, so `avg_tone`, `avg_goldstein` and `news_sentiment` are lightweight proxies derived from event notes and event intensity. This should be described as a limitation.

## Label construction

The label is:

```text
label = 1 if future six-month escalation score >= dyad-specific quantile threshold
label = 0 otherwise
```

The default threshold is the 75th percentile per dyad. This creates an operational high-escalation label and avoids falsely claiming to predict formal war declarations.

## Recommended model tuning

Start with the default settings. Then tune only a small number of meaningful parameters:

### Logistic Regression

- Use `class_weight='balanced'` for class imbalance.
- Try `C = [0.1, 1.0, 10.0]`.

### Random Forest

- Tune `n_estimators = [100, 200, 400]`.
- Tune `max_depth = [4, 6, 8, None]`.
- Tune `min_samples_leaf = [1, 3, 5]`.
- Keep `class_weight='balanced'`.

### LSTM

- Keep it small: `hidden_size = 8, 12, 24`.
- Use `sequence_length = 3 or 6` months.
- Use 3-10 epochs for fast coursework experiments.
- If it underperforms, report this honestly: deep learning can be less effective on small, sparse dyad-month data.

### Multi-Agent Debate

Tune weights in `src/agents/multi_agent_debate.py`:

- Increase escalation weight if recall is too low.
- Increase de-escalation weight if false positives are too high.
- Increase data-driven weight if Random Forest has the best ROC-AUC.
- Keep a Judge Agent to combine perspectives rather than selecting one head.

## Evaluation priorities

For this task, do not rely only on accuracy. Report:

- ROC-AUC: ranking high-risk vs low-risk cases.
- Brier Score: probability calibration.
- F1 / Precision / Recall: classification quality.
- Runtime: computational cost.
- Explanation Score: automatic, rule-based explanation coverage.
- Noise robustness: performance under perturbed inputs.

## Current prediction

The `predict_current_risk` script trains on labelled historical ACLED cases, builds current cases from the latest six months of ACLED data, and outputs six-month risk estimates. These predictions are not verifiable until the future horizon passes.
