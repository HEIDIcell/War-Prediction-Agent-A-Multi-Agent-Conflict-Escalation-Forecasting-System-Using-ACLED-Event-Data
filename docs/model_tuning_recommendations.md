# Model Tuning Recommendations

## 1. Check label balance first

Run:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/market_cases_sample.csv'); print(df['label'].value_counts()); print(df.groupby('dyad')['label'].agg(['count','sum','mean']))"
```

If there are too few positive labels, rebuild with a lower threshold:

```powershell
python -m src.data.build_acled_market_cases --raw-dir data/raw/acled --start-year 2016 --end-year 2025 --quantile 0.70 --no-worldbank
```

If there are too many positives, increase to `--quantile 0.80`.

## 2. Traditional models

Tune Random Forest in `src/models/traditional_models.py`:

- `n_estimators`: 100, 200, 400
- `max_depth`: 4, 6, 8, None
- `min_samples_leaf`: 1, 3, 5

Report Brier Score and ROC-AUC, not just Accuracy.

## 3. LSTM

Tune in `src/models/lstm_model.py`:

- `sequence_length`: 3 or 6 months
- `hidden_size`: 8, 16, 32
- `epochs`: 5, 10, 15
- `lr`: 0.001 or 0.005

If LSTM performs worse than Random Forest, this is acceptable and should be discussed: the dataset is small and sparse at dyad-month level.

## 4. Multi-agent Judge weights

Tune in `src/agents/multi_agent_debate.py`:

- More false negatives: increase EscalationAgent/DataDrivenAgent weight.
- Too many false positives: increase DeescalationAgent weight.
- Weak explanations: add more evidence terms from GDELT cache and ACLED notes.

## 5. Data enrichment order

Recommended order:

1. ACLED only: stable baseline.
2. ACLED + World Bank: annual context.
3. ACLED + GDELT: RAG/news explanation.
4. UCDP auxiliary validation: sanity-check only.
5. COW background check: report transparency only.
