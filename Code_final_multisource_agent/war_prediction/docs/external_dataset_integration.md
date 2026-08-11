# External Dataset Integration Notes

This project uses ACLED as the main supervised-learning dataset because the uploaded ACLED event-level CSV covers the recent period needed for current six-month forecasting. The additional datasets are integrated carefully according to temporal coverage.

## Main training source: ACLED

ACLED event records are aggregated into monthly US-target dyad cases. The label is an operational **future six-month high-escalation label**, derived from future ACLED event intensity. It is not a formal declaration-of-war label.

## GDELT: news sentiment and RAG explanation

GDELT is optional and used for the explanation/news layer rather than the primary label. Run:

```powershell
python -m src.data.build_gdelt_rag_cache --cases data/processed/market_cases_sample.csv --months-last 18 --max-records 15
```

This queries the public GDELT DOC API and writes:

```text
data/processed/gdelt_rag_cache.csv
data/processed/market_cases_with_gdelt.csv
```

The architecture experiments can use the GDELT-enriched file by replacing or copying it:

```powershell
copy data\processed\market_cases_with_gdelt.csv data\processed\market_cases_sample.csv
```

If the query fails, the ACLED-only system still runs.

## World Bank: annual GEO_CONTEXT

World Bank indicators are used when `--no-worldbank` is **not** supplied:

```powershell
python -m src.data.build_acled_market_cases --raw-dir data/raw/acled --start-year 2016 --end-year 2025
```

The code attempts to fetch annual indicators for GDP, military expenditure, armed forces personnel and trade as % of GDP. If values are missing, deterministic fallback proxies are used. For fully offline/reproducible experiments, use:

```powershell
python -m src.data.build_acled_market_cases --raw-dir data/raw/acled --start-year 2016 --end-year 2025 --no-worldbank
```

## UCDP: auxiliary validation only

UCDP is not used as the main training label because recent coverage does not perfectly align with ACLED 2016-2025 and because the project label is defined as future six-month escalation intensity. If you download UCDP GED CSV files, place them in:

```text
data/raw/ucdp/
```

Then run:

```powershell
python -m src.data.ucdp_auxiliary_validation --cases data/processed/market_cases_sample.csv
```

Outputs:

```text
data/processed/market_cases_with_ucdp_validation.csv
data/processed/ucdp_auxiliary_validation_summary.csv
```

This is a sanity-check, not a replacement label.

## COW NMC / MID: background and historical comparison

COW NMC and MID are important historical datasets, but their temporal coverage is not aligned with the ACLED 2016-2025 supervised task. They are therefore documented as background and future work. If you place COW CSV files in:

```text
data/raw/cow/
```

run:

```powershell
python -m src.data.cow_background_check
```

This writes:

```text
results/tables/cow_background_coverage.csv
```

Use this to show transparently why COW is not merged into the recent ACLED training labels.
