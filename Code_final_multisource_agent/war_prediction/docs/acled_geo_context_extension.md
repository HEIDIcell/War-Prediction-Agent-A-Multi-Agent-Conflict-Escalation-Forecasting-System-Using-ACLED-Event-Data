# ACLED + GEO_CONTEXT Extension

This extension adds a current-risk workflow to the War Prediction Agent project.

## Purpose

The original architecture-comparison experiment evaluates historical cases. This extension allows the system to train on past event data and predict the next-six-month escalation risk using the latest ACLED events.

## Data sources

- **ACLED event-level data**: user-downloaded CSV files containing event_date, country, event_type, sub_event_type, actor fields, fatalities, notes and source.
- **World Bank API**: optional enrichment for GDP, military expenditure, armed forces and trade indicators.
- **GEO_CONTEXT features**: capability ratio, diplomatic score, trade dependence, news sentiment proxy and recent conflict intensity.

## Features

| Feature | Implementation |
|---|---|
| conflict_count | ACLED Battles, Explosions/Remote violence, Violence against civilians |
| military_count | ACLED Battles and Explosions/Remote violence |
| threat_count | ACLED Strategic developments |
| cooperation_count | ACLED sub-event text containing agreement/peace/ceasefire |
| diplomatic_count | Strategic/diplomatic proxy from sub-event text |
| avg_tone / avg_goldstein | Compatibility fields derived from ACLED notes sentiment proxy |
| cinc_ratio | World Bank capability proxy or deterministic fallback |
| diplomatic_score | Cooperation/diplomacy minus conflict pressure |
| trade_dependence | World Bank trade percent GDP proxy or deterministic fallback |
| news_sentiment | Scaled tone/Goldstein sentiment proxy |
| recent_conflicts | Rolling 60-month conflict pressure |

## Label

The label is a future six-month high-escalation label:

```text
label = 1 if future_6m_escalation_score >= dyad-specific 75th percentile
label = 0 otherwise
```

This label estimates escalation intensity. It is not a verified statement that a state initiated a war.

## Commands

```powershell
python -m src.data.build_acled_market_cases --raw-dir data/raw/acled --start-year 2020 --end-year 2025
python -m src.experiments.run_architecture_comparison
python -m src.experiments.run_multi_head_agent_evaluation
python -m src.data.build_current_cases --raw-dir data/raw/acled --lookback-months 6
python -m src.experiments.predict_current_risk
```
