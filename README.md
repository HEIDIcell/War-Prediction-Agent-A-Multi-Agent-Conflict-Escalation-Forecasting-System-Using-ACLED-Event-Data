<<<<<<< HEAD
# War Prediction Agent: Multi-Agent Conflict Escalation Forecasting

This project implements an agent-based system for estimating the probability of **high conflict escalation involving the United States and a target state within the next six months**. It is designed for COMP3004/4105 Designing Intelligent Agents coursework.

The project does **not** make definitive claims that the United States will start a war. It produces experimental risk estimates from public event data.

## Coursework fit

- **Environment:** a geopolitical event environment constructed from ACLED event-level data. Each case is a US-target dyad-month.
- **Autonomous agents:** Sentiment, Escalation, De-escalation, Geo-context, Data-driven and Judge agents.
- **Research question:** does a multi-agent debate architecture improve conflict-risk forecasting performance and robustness compared with model-only and single-agent baselines?
- **Experiments:** architecture comparison, multi-head agent ablation/evaluation, feature-noise robustness, and current six-month risk prediction.

## Data

The code expects ACLED event-level CSV files in:

```text
data/raw/acled/
```

The uploaded coursework version can use:

```text
data/raw/acled/ACLED Data_2016_2026.csv
```

The provided ACLED file actually covers 2016-2025. If you submit the code publicly, check ACLED's licence terms before uploading the raw CSV. For private Moodle submission, include it only if allowed by your data-use conditions.

## Installation

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch is slow to install, install core packages first:

```powershell
python -m pip install pandas numpy scikit-learn matplotlib requests
python -m pip install torch
```

## Step 1: Build ACLED market cases

Use all available years in the ACLED CSV:

```powershell
python -m src.data.build_acled_market_cases --raw-dir data/raw/acled --no-worldbank
```

Or specify the actual available range:

```powershell
python -m src.data.build_acled_market_cases --raw-dir data/raw/acled --start-year 2016 --end-year 2025 --no-worldbank
```

Outputs:

```text
data/processed/market_cases_sample.csv
data/processed/acled_market_cases_summary.csv
```

Check the dataset:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/market_cases_sample.csv'); print(df.shape); print(df['month'].min(), df['month'].max()); print(df['label'].value_counts()); print(df.groupby('dyad')['label'].agg(['count','sum','mean']))"
```

## Step 2: Architecture comparison experiment

```powershell
python -m src.experiments.run_architecture_comparison
```

Compares:

1. Logistic Regression
2. Random Forest
3. LSTM sequence model
4. Single RAG-style Agent
5. Multi-Agent Debate system

Outputs:

```text
results/tables/architecture_comparison_full.csv
results/tables/predictions_by_system.csv
results/figures/architecture_metrics_bar.png
results/figures/brier_score_comparison.png
results/figures/roc_auc_comparison.png
results/figures/explanation_score_comparison.png
results/figures/runtime_comparison.png
```

## Step 3: Multi-head agent evaluation

```powershell
python -m src.experiments.run_multi_head_agent_evaluation
```

Outputs:

```text
results/tables/multi_head_agent_evaluation.csv
```

This evaluates individual agent heads against the full Judge system.

## Step 4: Noise robustness experiment

```powershell
python -m src.experiments.run_noise_robustness
```

Outputs:

```text
results/tables/noise_robustness_raw.csv
results/tables/noise_robustness_summary.csv
results/figures/noise_robustness_brier.png
results/figures/noise_robustness_roc_auc.png
```

## Step 5: Current six-month risk prediction

Build current cases from the latest six months in the ACLED file:

```powershell
python -m src.data.build_current_cases --raw-dir data/raw/acled --lookback-months 6 --no-worldbank
```

Predict future six-month risk:

```powershell
python -m src.experiments.predict_current_risk
```

Output:

```text
results/reports/current_risk_predictions.csv
```

Interpretation example:

```text
US-Iran, Multi-Agent_Debate, predicted_probability=0.81, risk_level=Very high
```

This means the system estimates a very high probability of **high conflict escalation** in the next six months. It does not mean that a war will definitely occur.

## Recommended report wording

Use this wording rather than saying "predicts whether the US will start a war":

> The system estimates the probability of high conflict escalation involving the United States and a target state within a six-month horizon, based on historical public event data.

## Main limitations

- ACLED is an event dataset, not a formal war-declaration dataset.
- The label is an operational future-escalation label derived from future ACLED event intensity.
- The data file used here contains Russia, China and Iran, but not North Korea event data. The code automatically drops missing dyads.
- Current predictions cannot be verified until the six-month horizon has passed.
- Strategic or policy advice is intentionally not generated.

## Optional external dataset enrichment

The stable coursework pipeline uses the uploaded ACLED CSV. The project also contains optional integration for the four additional datasets discussed in the design notes.

### A. World Bank GEO_CONTEXT

Use World Bank annual indicators by omitting `--no-worldbank`:

```powershell
python -m src.data.build_acled_market_cases --raw-dir data/raw/acled --start-year 2016 --end-year 2025
```

This attempts to add GDP, military expenditure, armed forces personnel and trade % GDP context. If the API is unavailable or indicators are missing, fallback proxies are used automatically. For reproducible offline runs, keep using `--no-worldbank`.

### B. GDELT news/RAG enrichment

After building ACLED cases, optionally query GDELT DOC for recent US-target news evidence:

```powershell
python -m src.data.build_gdelt_rag_cache --cases data/processed/market_cases_sample.csv --months-last 18 --max-records 15
copy data\processed\market_cases_with_gdelt.csv data\processed\market_cases_sample.csv
python -m src.experiments.run_architecture_comparison
```

This adds optional columns such as `gdelt_article_count`, `gdelt_avg_tone` and `gdelt_negative_share`. The Single-Agent RAG retriever will also include cached GDELT article titles in explanations when available.

### C. UCDP auxiliary validation

UCDP is not used as the main label. To run an optional sanity-check, place downloaded UCDP GED CSV files in `data/raw/ucdp/` and run:

```powershell
python -m src.data.ucdp_auxiliary_validation --cases data/processed/market_cases_sample.csv
```

Outputs:

```text
data/processed/market_cases_with_ucdp_validation.csv
data/processed/ucdp_auxiliary_validation_summary.csv
```

### D. COW NMC / MID background check

COW NMC/MID are background datasets because their coverage does not align with the recent 2016-2025 ACLED supervised task. To document coverage, place COW CSV files in `data/raw/cow/` and run:

```powershell
python -m src.data.cow_background_check
```

Output:

```text
results/tables/cow_background_coverage.csv
```

## Recommended complete run

```powershell
# 1. Stable ACLED-only dataset
python -m src.data.build_acled_market_cases --raw-dir data/raw/acled --start-year 2016 --end-year 2025 --no-worldbank

# 2. Optional GDELT enrichment for recent news/RAG explanations
python -m src.data.build_gdelt_rag_cache --cases data/processed/market_cases_sample.csv --months-last 18 --max-records 15
copy data\processed\market_cases_with_gdelt.csv data\processed\market_cases_sample.csv

# 3. Main experiments
python -m src.experiments.run_architecture_comparison
python -m src.experiments.run_multi_head_agent_evaluation
python -m src.experiments.run_noise_robustness

# 4. Current six-month prediction
python -m src.data.build_current_cases --raw-dir data/raw/acled --lookback-months 6 --no-worldbank
python -m src.experiments.predict_current_risk
```
=======
# War-Prediction-Agent-A-Multi-Agent-Conflict-Escalation-Forecasting-System-Using-ACLED-Event-Data
A multi-agent conflict escalation forecasting system using ACLED event data (2016-2025) for US-China, US-Iran, and US-Russia dyads. Compares Logistic Regression, Random Forest, LSTM, and a six-agent Multi-Agent Debate architecture (Sentiment, Escalation, De-escalation, Geo-context, Data, Judge), reaching 0.905 ROC-AUC with tested noise robustness. 
>>>>>>> 0279aba339ea9ea5f7a3bbe14996b5afb28a2530
