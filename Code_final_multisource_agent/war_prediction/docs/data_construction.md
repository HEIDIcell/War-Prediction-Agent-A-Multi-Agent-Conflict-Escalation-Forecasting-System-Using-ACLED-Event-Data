# Data Construction: GDELT + COW MID

This project includes an integrated data-construction script:

```powershell
python -m src.data.build_real_market_cases --start-year 2000 --end-year 2012
```

The command builds a 13-year dataset covering 2000-2012 inclusive. For four dyads — US-China, US-Russia, US-Iran and US-NorthKorea — it converts public GDELT 1.0 event files into monthly features and uses COW Dyadic MID as the future six-month label source.

## Output files

```text
data/processed/gdelt_monthly_features.csv
data/processed/market_cases_sample.csv
data/processed/market_cases_summary.csv
```

`market_cases_sample.csv` is the file used by the architecture comparison experiment.

## Required output columns

```text
case_id, dyad, month,
conflict_count, military_count, threat_count,
cooperation_count, diplomatic_count,
avg_tone, avg_goldstein,
model_probability, label
```

## Why fixed local data instead of live API calls?

The experiment uses a fixed local CSV after data construction. This makes the results reproducible and avoids changing outputs caused by network conditions or updates to external services.
