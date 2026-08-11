"""Build current prediction cases from the latest months in local ACLED event data.

This produces rows with label=-1 because future outcomes are unknown. The rows
can be consumed by src.experiments.predict_current_risk.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data.build_acled_market_cases import load_acled_files, build_monthly_features, add_model_probability, DYADS


def build_current_cases(raw_dir: str, out_path: str, lookback_months: int = 6, use_worldbank: bool = True):
    acled = load_acled_files(raw_dir)
    monthly = build_monthly_features(acled, use_worldbank=use_worldbank)
    max_month = monthly["month_dt"].max()
    if pd.isna(max_month):
        raise ValueError("Could not identify latest ACLED month.")
    start_month = (pd.Timestamp(max_month) - pd.DateOffset(months=lookback_months-1)).replace(day=1)
    window = monthly[(monthly["month_dt"] >= start_month) & (monthly["month_dt"] <= max_month)].copy()
    if window.empty:
        raise ValueError("No data in current lookback window.")

    rows = []
    for dyad, g in window.groupby("dyad"):
        g = g.sort_values("month_dt")
        rows.append({
            "case_id": f"CURRENT_{dyad.replace('-', '_')}",
            "dyad": dyad,
            "month": pd.Timestamp(max_month).strftime("%Y-%m"),
            "prediction_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "lookback_months": lookback_months,
            "conflict_count": g["conflict_count"].sum(),
            "military_count": g["military_count"].sum(),
            "threat_count": g["threat_count"].sum(),
            "cooperation_count": g["cooperation_count"].sum(),
            "diplomatic_count": g["diplomatic_count"].sum(),
            "avg_tone": g["avg_tone"].mean(),
            "avg_goldstein": g["avg_goldstein"].mean(),
            "cinc_ratio": g["cinc_ratio"].iloc[-1],
            "diplomatic_score": g["diplomatic_score"].mean(),
            "trade_dependence": g["trade_dependence"].iloc[-1],
            "us_gdp_current_usd": g.get("us_gdp_current_usd", pd.Series([0.0])).iloc[-1],
            "target_gdp_current_usd": g.get("target_gdp_current_usd", pd.Series([0.0])).iloc[-1],
            "us_military_expenditure_usd": g.get("us_military_expenditure_usd", pd.Series([0.0])).iloc[-1],
            "target_military_expenditure_usd": g.get("target_military_expenditure_usd", pd.Series([0.0])).iloc[-1],
            "us_trade_percent_gdp": g.get("us_trade_percent_gdp", pd.Series([0.0])).iloc[-1],
            "target_trade_percent_gdp": g.get("target_trade_percent_gdp", pd.Series([0.0])).iloc[-1],
            "news_sentiment": g["news_sentiment"].mean(),
            "recent_conflicts": g["recent_conflicts"].iloc[-1],
            "fatalities_sum": g["fatalities_sum"].sum(),
            "us_actor_events": g["us_actor_events"].sum(),
            "label": -1,
        })
    out = pd.DataFrame(rows)
    out = add_model_probability(out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path}: {out.shape}")
    print(out[["dyad", "month", "conflict_count", "military_count", "threat_count", "model_probability"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/acled")
    ap.add_argument("--out", default="data/processed/current_cases.csv")
    ap.add_argument("--lookback-months", type=int, default=6)
    ap.add_argument("--no-worldbank", action="store_true")
    args = ap.parse_args()
    build_current_cases(args.raw_dir, args.out, args.lookback_months, use_worldbank=not args.no_worldbank)

if __name__ == "__main__":
    main()
