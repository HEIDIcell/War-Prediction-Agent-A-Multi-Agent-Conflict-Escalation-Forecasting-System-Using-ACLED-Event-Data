"""
Build market_cases_sample.csv from locally downloaded ACLED event-level CSV files.

Recommended ACLED download:
- Event data, CSV format
- Countries: United States, China, Russia, Iran, North Korea
- Date range: as much recent data as possible, e.g. 2020-01-01 to latest
- Event types: all

The script turns ACLED event rows into monthly US-target dyad cases and creates
a future-six-month escalation label. The label is not a verified declaration of
war; it is an operational high-escalation label derived from future event intensity.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data.context_features import add_context_features, COUNTRY_NAME_BY_DYAD, lexical_sentiment_from_notes

DYADS = ["US-China", "US-Russia", "US-Iran", "US-NorthKorea"]
COUNTRY_ALIASES = {
    "US-China": ["China"],
    "US-Russia": ["Russia"],
    "US-Iran": ["Iran"],
    "US-NorthKorea": ["North Korea", "Korea, North", "Democratic People's Republic of Korea"],
}

CONFLICT_TYPES = {"Battles", "Explosions/Remote violence", "Violence against civilians"}
MILITARY_TYPES = {"Battles", "Explosions/Remote violence"}
STRATEGIC_TYPES = {"Strategic developments"}
COOPERATION_SUBSTRINGS = ["agreement", "change to group/activity", "peace", "headquarters or base established"]
US_PATTERNS = ["United States", "US Forces", "U.S.", "USA", "American"]


def load_acled_files(raw_dir: str | Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(raw_dir) / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No ACLED CSV files found in {raw_dir}. Put downloaded ACLED event-data CSV files there."
        )
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
            print(f"[INFO] Loaded ACLED file: {f}")
        except UnicodeDecodeError:
            frames.append(pd.read_csv(f, encoding="latin1", low_memory=False))
            print(f"[INFO] Loaded ACLED file with latin1 encoding: {f}")
    df = pd.concat(frames, ignore_index=True)
    if "event_date" not in df.columns:
        raise ValueError("ACLED file must contain an event_date column.")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["event_date"]).copy()
    df["month_dt"] = df["event_date"].values.astype("datetime64[M]")
    df["month"] = pd.to_datetime(df["month_dt"]).dt.strftime("%Y-%m")
    for col in ["country", "event_type", "sub_event_type", "actor1", "actor2", "assoc_actor_1", "assoc_actor_2", "notes"]:
        if col not in df.columns:
            df[col] = ""
    if "fatalities" not in df.columns:
        df["fatalities"] = 0
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
    return df


def country_mask(df: pd.DataFrame, aliases: List[str]) -> pd.Series:
    country = df["country"].fillna("").astype(str)
    mask = pd.Series(False, index=df.index)
    for alias in aliases:
        mask |= country.str.lower().eq(alias.lower())
    return mask


def contains_any(series: pd.Series, patterns: List[str]) -> pd.Series:
    s = series.fillna("").astype(str)
    mask = pd.Series(False, index=series.index)
    for pat in patterns:
        mask |= s.str.contains(pat, case=False, regex=False, na=False)
    return mask


def build_monthly_features(acled: pd.DataFrame, start_year: int | None = None, end_year: int | None = None,
                           use_worldbank: bool = True) -> pd.DataFrame:
    if start_year is not None:
        acled = acled[acled["event_date"].dt.year >= start_year]
    if end_year is not None:
        acled = acled[acled["event_date"].dt.year <= end_year]
    if acled.empty:
        raise ValueError("No ACLED rows remain after date filtering.")

    all_months = pd.date_range(acled["month_dt"].min(), acled["month_dt"].max(), freq="MS")
    rows = []
    for dyad in DYADS:
        aliases = COUNTRY_ALIASES[dyad]
        ddf = acled[country_mask(acled, aliases)].copy()
        for month_dt in all_months:
            mdf = ddf[ddf["month_dt"] == month_dt]
            event_type = mdf["event_type"].fillna("").astype(str)
            sub_event = mdf["sub_event_type"].fillna("").astype(str)
            actors_text = (
                mdf["actor1"].fillna("").astype(str) + " " +
                mdf["actor2"].fillna("").astype(str) + " " +
                mdf["assoc_actor_1"].fillna("").astype(str) + " " +
                mdf["assoc_actor_2"].fillna("").astype(str)
            )
            conflict_count = int(event_type.isin(CONFLICT_TYPES).sum())
            military_count = int(event_type.isin(MILITARY_TYPES).sum())
            threat_count = int(event_type.isin(STRATEGIC_TYPES).sum())
            # ACLED has limited direct diplomacy/cooperation coverage; use Strategic developments and notes as a weak proxy.
            cooperation_count = int(sub_event.str.contains("agreement|peace|ceasefire|non-violent", case=False, regex=True, na=False).sum())
            diplomatic_count = int(sub_event.str.contains("agreement|peace|ceasefire|headquarters|change", case=False, regex=True, na=False).sum())
            us_actor_events = int(contains_any(actors_text, US_PATTERNS).sum())
            fatalities_sum = float(mdf["fatalities"].sum()) if len(mdf) else 0.0
            notes_sent = lexical_sentiment_from_notes(mdf.get("notes", pd.Series(dtype=str)).astype(str).tolist())
            # Map ACLED notes sentiment proxy into old tone/Goldstein-style fields for compatibility.
            avg_tone = float(10.0 * notes_sent)
            avg_goldstein = float(5.0 * notes_sent - 0.01 * conflict_count - 0.02 * fatalities_sum)
            rows.append({
                "dyad": dyad,
                "month": pd.to_datetime(month_dt).strftime("%Y-%m"),
                "month_dt": pd.to_datetime(month_dt),
                "total_event_count": int(len(mdf)),
                "conflict_count": conflict_count,
                "military_count": military_count,
                "threat_count": threat_count,
                "cooperation_count": cooperation_count,
                "diplomatic_count": diplomatic_count,
                "avg_tone": avg_tone,
                "avg_goldstein": avg_goldstein,
                "fatalities_sum": fatalities_sum,
                "us_actor_events": us_actor_events,
            })
    monthly = pd.DataFrame(rows).sort_values(["dyad", "month_dt"]).reset_index(drop=True)
    # Drop target dyads that are not actually present in the downloaded ACLED file.
    # This prevents an all-zero target, such as US-NorthKorea when no North Korea rows
    # were downloaded, from producing meaningless labels.
    dyad_volume = monthly.groupby("dyad")["total_event_count"].sum()
    valid_dyads = dyad_volume[dyad_volume > 0].index.tolist()
    dropped = sorted(set(monthly["dyad"].unique()) - set(valid_dyads))
    if dropped:
        print(f"[WARNING] Dropping dyads with no ACLED events in the input file: {dropped}")
    monthly = monthly[monthly["dyad"].isin(valid_dyads)].copy()
    monthly = add_context_features(monthly, use_worldbank=use_worldbank)
    return monthly


def add_future_escalation_label(monthly: pd.DataFrame, horizon_months: int = 6, quantile: float = 0.75,
                                label_mode: str = "escalation") -> pd.DataFrame:
    """Create future 6-month labels.

    label_mode='escalation': label high future conflict intensity.
    label_mode='us_involvement': label future months with US actor involvement + high conflict.
    """
    df = monthly.sort_values(["dyad", "month_dt"]).copy()
    escalation_now = (
        1.00 * df["conflict_count"].fillna(0) +
        1.50 * df["military_count"].fillna(0) +
        1.20 * df["threat_count"].fillna(0) +
        0.02 * df["fatalities_sum"].fillna(0) -
        0.40 * df["cooperation_count"].fillna(0) -
        0.30 * df["diplomatic_count"].fillna(0)
    )
    if label_mode == "us_involvement":
        escalation_now = escalation_now + 2.0 * df["us_actor_events"].fillna(0)
    df["escalation_score_current"] = escalation_now.clip(lower=0)
    df["future_6m_escalation_score"] = (
        df.groupby("dyad")["escalation_score_current"]
        .transform(lambda s: s.shift(-1).rolling(window=horizon_months, min_periods=horizon_months).sum().shift(-(horizon_months-1)))
    )
    # Drop final rows without a full future window.
    trainable = df.dropna(subset=["future_6m_escalation_score"]).copy()
    thresholds = {}
    trainable["label"] = 0
    for dyad, group in trainable.groupby("dyad"):
        th = float(group["future_6m_escalation_score"].quantile(quantile))
        # If there is no variation or no escalation signal for a dyad, mark it as low-risk
        # rather than labelling every month positive.
        if th <= 0 or group["future_6m_escalation_score"].nunique() <= 1:
            thresholds[dyad] = 0.0
            trainable.loc[group.index, "label"] = 0
        else:
            thresholds[dyad] = th
            trainable.loc[group.index, "label"] = (group["future_6m_escalation_score"] >= th).astype(int).values
    trainable["label"] = trainable["label"].astype(int)
    trainable["label_threshold"] = trainable["dyad"].map(thresholds)
    return trainable


def add_model_probability(df: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent heuristic probability before the full experiment models run."""
    out = df.copy()
    z = (
        0.05*out["conflict_count"].fillna(0) + 0.08*out["military_count"].fillna(0) +
        0.08*out["threat_count"].fillna(0) - 0.04*out["cooperation_count"].fillna(0) -
        0.03*out["diplomatic_count"].fillna(0) - 0.25*out["news_sentiment"].fillna(0) +
        0.70*out["recent_conflicts"].fillna(0) + 0.05*out["cinc_ratio"].fillna(1.0) - 1.0
    )
    out["model_probability"] = 1.0 / (1.0 + np.exp(-z))
    out["model_probability"] = out["model_probability"].clip(0.001, 0.999)
    return out


def build_cases(raw_dir: str, out_path: str, start_year: int | None, end_year: int | None,
                horizon_months: int, quantile: float, label_mode: str, use_worldbank: bool) -> pd.DataFrame:
    acled = load_acled_files(raw_dir)
    monthly = build_monthly_features(acled, start_year=start_year, end_year=end_year, use_worldbank=use_worldbank)
    labelled = add_future_escalation_label(monthly, horizon_months=horizon_months, quantile=quantile, label_mode=label_mode)
    labelled = add_model_probability(labelled)
    labelled = labelled.sort_values(["dyad", "month_dt"]).reset_index(drop=True)
    labelled.insert(0, "case_id", [f"ACLED_CASE_{i:05d}" for i in range(len(labelled))])
    # A transparent chronological split for the report: the experiment script also
    # falls back to temporal splitting if this column is absent.
    years = pd.to_datetime(labelled["month"] + "-01").dt.year
    labelled["split"] = np.where(years <= 2022, "train", np.where(years == 2023, "validation", "test"))
    required_order = [
        "case_id", "dyad", "month", "split", "total_event_count",
        "conflict_count", "military_count", "threat_count",
        "cooperation_count", "diplomatic_count", "avg_tone", "avg_goldstein",
        "cinc_ratio", "diplomatic_score", "trade_dependence", "news_sentiment", "recent_conflicts",
        "us_gdp_current_usd", "target_gdp_current_usd",
        "us_military_expenditure_usd", "target_military_expenditure_usd",
        "us_trade_percent_gdp", "target_trade_percent_gdp",
        "fatalities_sum", "us_actor_events", "model_probability", "label",
        "future_6m_escalation_score", "label_threshold",
    ]
    keep = [c for c in required_order if c in labelled.columns]
    out = labelled[keep].copy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    summary = out.groupby("dyad")["label"].agg(["count", "sum", "mean"]).reset_index()
    summary.to_csv(Path(out_path).parent / "acled_market_cases_summary.csv", index=False)
    print(f"[INFO] Wrote {out_path}: {out.shape}")
    print("[INFO] Label distribution:")
    print(out["label"].value_counts())
    print("[INFO] By dyad:")
    print(summary)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/acled")
    ap.add_argument("--out", default="data/processed/market_cases_sample.csv")
    ap.add_argument("--start-year", type=int, default=None)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--horizon-months", type=int, default=6)
    ap.add_argument("--quantile", type=float, default=0.75)
    ap.add_argument("--label-mode", choices=["escalation", "us_involvement"], default="escalation")
    ap.add_argument("--no-worldbank", action="store_true", help="Disable online World Bank API enrichment and use fallback proxies.")
    args = ap.parse_args()
    build_cases(
        raw_dir=args.raw_dir, out_path=args.out, start_year=args.start_year, end_year=args.end_year,
        horizon_months=args.horizon_months, quantile=args.quantile, label_mode=args.label_mode,
        use_worldbank=not args.no_worldbank,
    )

if __name__ == "__main__":
    main()
