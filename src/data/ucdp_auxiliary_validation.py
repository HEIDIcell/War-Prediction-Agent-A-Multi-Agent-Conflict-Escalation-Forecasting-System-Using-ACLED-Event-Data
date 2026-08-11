"""Auxiliary validation against UCDP GED-style CSV files.

UCDP is not used as the main label because its recent release coverage does not
perfectly match the ACLED 2016-2025 training pipeline. This script is optional:
place one or more UCDP GED CSV files in data/raw/ucdp/ and run:

    python -m src.data.ucdp_auxiliary_validation --cases data/processed/market_cases_sample.csv

It aggregates UCDP events by target-country month and joins the counts to the
ACLED-derived dyad-month cases. The output is a sanity-check table, not a new
training target.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

TARGETS = {
    "US-China": ["China"],
    "US-Russia": ["Russia", "Russian Federation"],
    "US-Iran": ["Iran"],
    "US-NorthKorea": ["North Korea", "Korea, North"],
}


def load_ucdp(raw_dir: str) -> pd.DataFrame:
    paths = list(Path(raw_dir).glob("*.csv"))
    if not paths:
        print(f"[WARNING] No UCDP CSV files found in {raw_dir}. Auxiliary validation skipped.")
        return pd.DataFrame()
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p, low_memory=False))
        except Exception as exc:
            print(f"[WARNING] Could not read {p}: {exc}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def normalise_ucdp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # Try common UCDP GED date/country/fatality column names.
    date_col = next((c for c in ["date_start", "date", "event_date", "start_date"] if c in out.columns), None)
    country_col = next((c for c in ["country", "country_name", "country_id", "side_a", "side_b"] if c in out.columns), None)
    fatal_col = next((c for c in ["best", "deaths_a", "deaths_b", "deaths_civilians", "fatalities"] if c in out.columns), None)
    if date_col is None or country_col is None:
        raise ValueError("UCDP CSV does not contain recognisable date/country columns.")
    out["event_date"] = pd.to_datetime(out[date_col], errors="coerce")
    out["month"] = out["event_date"].dt.strftime("%Y-%m")
    out["country_text"] = out[country_col].astype(str)
    out["ucdp_fatalities"] = pd.to_numeric(out[fatal_col], errors="coerce").fillna(0) if fatal_col else 0.0
    return out.dropna(subset=["month"])


def join_validation(cases_path: str, raw_dir: str, out_path: str) -> pd.DataFrame:
    cases = pd.read_csv(cases_path)
    u = normalise_ucdp(load_ucdp(raw_dir))
    if u.empty:
        cases["ucdp_validation_events"] = 0
        cases["ucdp_validation_fatalities"] = 0.0
        cases.to_csv(out_path, index=False)
        return cases
    rows = []
    for dyad, names in TARGETS.items():
        mask = False
        for name in names:
            mask = mask | u["country_text"].str.contains(re.escape(name), case=False, na=False)
        g = u[mask].groupby("month").agg(
            ucdp_validation_events=("month", "size"),
            ucdp_validation_fatalities=("ucdp_fatalities", "sum"),
        ).reset_index()
        g["dyad"] = dyad
        rows.append(g)
    val = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = cases.merge(val, on=["dyad", "month"], how="left")
    out[["ucdp_validation_events", "ucdp_validation_fatalities"]] = out[
        ["ucdp_validation_events", "ucdp_validation_fatalities"]
    ].fillna(0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    # Basic sanity-check summary.
    summary = out.groupby("dyad").agg(
        acled_positive_rate=("label", "mean"),
        ucdp_event_months=("ucdp_validation_events", lambda s: int((s > 0).sum())),
        ucdp_total_fatalities=("ucdp_validation_fatalities", "sum"),
    ).reset_index()
    summary_path = str(Path(out_path).with_name("ucdp_auxiliary_validation_summary.csv"))
    summary.to_csv(summary_path, index=False)
    print(f"[INFO] Wrote {out_path}: {out.shape}")
    print(f"[INFO] Wrote {summary_path}: {summary.shape}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/processed/market_cases_sample.csv")
    ap.add_argument("--raw-dir", default="data/raw/ucdp")
    ap.add_argument("--out", default="data/processed/market_cases_with_ucdp_validation.csv")
    args = ap.parse_args()
    join_validation(args.cases, args.raw_dir, args.out)


if __name__ == "__main__":
    main()
