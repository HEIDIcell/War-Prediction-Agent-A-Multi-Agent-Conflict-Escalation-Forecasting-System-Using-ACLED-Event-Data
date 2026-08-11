"""
Geo-context feature helpers for the War Prediction Agent.

This module is intentionally robust: it can enrich cases with World Bank
capability/economic indicators when internet access is available, but it also
falls back to deterministic proxy/default values so that coursework experiments
remain reproducible.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

TARGET_COUNTRY_CODES = {
    "US-China": "CHN",
    "US-Russia": "RUS",
    "US-Iran": "IRN",
    "US-NorthKorea": "PRK",
}

COUNTRY_NAME_BY_DYAD = {
    "US-China": "China",
    "US-Russia": "Russia",
    "US-Iran": "Iran",
    "US-NorthKorea": "North Korea",
}

# World Bank indicator codes. These are used only when online access is available.
WB_INDICATORS = {
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "military_expenditure_usd": "MS.MIL.XPND.CD",
    "armed_forces_total": "MS.MIL.TOTL.P1",
    "trade_percent_gdp": "NE.TRD.GNFS.ZS",
}

# Deterministic proxy values used when online indicators are missing.
# They are intentionally approximate and should be described as fallback proxies.
FALLBACK_CONTEXT = {
    "US-China":      {"cinc_ratio": 0.85, "trade_dependence": 0.58},
    "US-Russia":     {"cinc_ratio": 1.25, "trade_dependence": 0.35},
    "US-Iran":       {"cinc_ratio": 2.20, "trade_dependence": 0.28},
    "US-NorthKorea": {"cinc_ratio": 2.60, "trade_dependence": 0.18},
}

NEGATIVE_TERMS = [
    "attack", "clash", "strike", "missile", "shell", "bomb", "arrest", "killed",
    "death", "fatal", "violence", "military", "sanction", "threat", "tension",
]
POSITIVE_TERMS = [
    "talk", "meeting", "agreement", "cooperation", "ceasefire", "peace", "diplomatic",
    "dialogue", "negotiation", "aid", "visit", "summit",
]


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def lexical_sentiment_from_notes(notes: Iterable[str]) -> float:
    """Return a simple [-1, 1] sentiment proxy from ACLED notes.

    ACLED is not a sentiment dataset, so this is only a lightweight fallback
    for the news_sentiment feature when GDELT DOC sentiment is not used.
    """
    neg = 0
    pos = 0
    total = 0
    for text in notes:
        if not isinstance(text, str):
            continue
        lower = text.lower()
        total += 1
        neg += sum(1 for w in NEGATIVE_TERMS if w in lower)
        pos += sum(1 for w in POSITIVE_TERMS if w in lower)
    if total == 0:
        return 0.0
    return float(np.clip((pos - neg) / max(pos + neg, 1), -1.0, 1.0))


def world_bank_series(country_code: str, indicator: str, start_year: int, end_year: int,
                      cache_dir: str | Path = "data/raw/worldbank") -> Dict[int, float]:
    """Fetch a World Bank indicator as {year: value}.

    Uses the public World Bank API directly. If requests is unavailable or the
    API fails, returns an empty dict and the caller will fall back gracefully.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{country_code}_{indicator}_{start_year}_{end_year}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return {int(k): float(v) for k, v in data.items() if v is not None}
        except Exception:
            pass

    if requests is None:
        return {}
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
        f"?format=json&per_page=20000&date={start_year}:{end_year}"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        out = {}
        for row in rows:
            year = int(row.get("date"))
            value = row.get("value")
            if value is not None:
                out[year] = float(value)
        cache_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
        time.sleep(0.1)
        return out
    except Exception:
        return {}


def build_worldbank_context(dyads: Iterable[str], years: Iterable[int]) -> pd.DataFrame:
    """Build annual dyad-level context features from World Bank indicators.

    cinc_ratio is implemented as a simple capability proxy rather than the
    official COW CINC index for post-2016 years: it combines GDP, military
    expenditure and armed forces personnel. Higher values mean the US capability
    proxy is larger relative to the target state's proxy.
    """
    years = sorted(set(int(y) for y in years))
    if not years:
        return pd.DataFrame()
    start_year, end_year = min(years), max(years)

    # Fetch US once.
    us_data = {name: world_bank_series("USA", ind, start_year, end_year)
               for name, ind in WB_INDICATORS.items()}

    rows = []
    for dyad in dyads:
        target_code = TARGET_COUNTRY_CODES.get(dyad)
        target_data = {name: world_bank_series(target_code, ind, start_year, end_year)
                       for name, ind in WB_INDICATORS.items()} if target_code else {}
        for year in years:
            us_gdp = safe_float(us_data.get("gdp_current_usd", {}).get(year))
            us_mil = safe_float(us_data.get("military_expenditure_usd", {}).get(year))
            us_troops = safe_float(us_data.get("armed_forces_total", {}).get(year))
            t_gdp = safe_float(target_data.get("gdp_current_usd", {}).get(year))
            t_mil = safe_float(target_data.get("military_expenditure_usd", {}).get(year))
            t_troops = safe_float(target_data.get("armed_forces_total", {}).get(year))
            us_trade = safe_float(us_data.get("trade_percent_gdp", {}).get(year))
            t_trade = safe_float(target_data.get("trade_percent_gdp", {}).get(year))

            # Capability proxy: average of log-scaled GDP, military expenditure and troops.
            def cap(gdp, mil, troops):
                vals = [gdp, mil, troops]
                vals = [np.log1p(v) for v in vals if not pd.isna(v) and v > 0]
                return float(np.mean(vals)) if vals else np.nan
            us_cap, target_cap = cap(us_gdp, us_mil, us_troops), cap(t_gdp, t_mil, t_troops)
            if pd.isna(us_cap) or pd.isna(target_cap) or target_cap <= 0:
                cinc_ratio = FALLBACK_CONTEXT.get(dyad, {}).get("cinc_ratio", 1.0)
            else:
                cinc_ratio = float(np.clip(us_cap / target_cap, 0.05, 5.0))

            if pd.isna(us_trade) and pd.isna(t_trade):
                trade_dependence = FALLBACK_CONTEXT.get(dyad, {}).get("trade_dependence", 0.35)
            else:
                trade_dependence = float(np.nanmean([us_trade, t_trade]) / 100.0)
                trade_dependence = float(np.clip(trade_dependence, 0.0, 2.0))

            rows.append({
                "dyad": dyad,
                "year": year,
                "cinc_ratio": cinc_ratio,
                "trade_dependence": trade_dependence,
            })
    return pd.DataFrame(rows)


def add_context_features(monthly_df: pd.DataFrame, use_worldbank: bool = True) -> pd.DataFrame:
    """Add GEO_CONTEXT features to a monthly dyad dataframe.

    Required existing columns: dyad, month_dt, conflict_count, cooperation_count,
    diplomatic_count, avg_tone, avg_goldstein. Missing values are handled.
    """
    df = monthly_df.copy()
    if "month_dt" not in df.columns:
        df["month_dt"] = pd.to_datetime(df["month"].astype(str) + "-01", errors="coerce")
    df["year"] = df["month_dt"].dt.year.astype(int)

    # local/proxy context first
    df["cinc_ratio"] = df["dyad"].map(lambda d: FALLBACK_CONTEXT.get(d, {}).get("cinc_ratio", 1.0))
    df["trade_dependence"] = df["dyad"].map(lambda d: FALLBACK_CONTEXT.get(d, {}).get("trade_dependence", 0.35))

    if use_worldbank:
        wb = build_worldbank_context(df["dyad"].unique(), df["year"].unique())
        if not wb.empty:
            df = df.drop(columns=["cinc_ratio", "trade_dependence"], errors="ignore").merge(
                wb, on=["dyad", "year"], how="left"
            )
            df["cinc_ratio"] = df["cinc_ratio"].fillna(df["dyad"].map(lambda d: FALLBACK_CONTEXT.get(d, {}).get("cinc_ratio", 1.0)))
            df["trade_dependence"] = df["trade_dependence"].fillna(df["dyad"].map(lambda d: FALLBACK_CONTEXT.get(d, {}).get("trade_dependence", 0.35)))

    # Diplomatic score: positive cooperation/diplomacy minus conflict pressure.
    denom = (df["cooperation_count"].fillna(0) + df["diplomatic_count"].fillna(0) +
             df["conflict_count"].fillna(0) + 1.0)
    df["diplomatic_score"] = (
        (df["cooperation_count"].fillna(0) + 1.2 * df["diplomatic_count"].fillna(0) - df["conflict_count"].fillna(0)) / denom
    ).clip(-1.0, 1.0)

    # News sentiment: scaled average of tone and Goldstein; positive means more cooperative.
    df["news_sentiment"] = (
        0.6 * np.tanh(df["avg_tone"].fillna(0) / 10.0) +
        0.4 * np.tanh(df["avg_goldstein"].fillna(0) / 10.0)
    ).clip(-1.0, 1.0)

    # Recent conflicts: rolling 60-month conflict pressure within each dyad.
    df = df.sort_values(["dyad", "month_dt"]).reset_index(drop=True)
    pressure = df["conflict_count"].fillna(0) + df.get("military_count", 0) + df.get("threat_count", 0)
    df["_pressure"] = pressure
    df["recent_conflicts"] = (
        df.groupby("dyad")["_pressure"]
        .transform(lambda s: s.shift(1).rolling(window=60, min_periods=1).mean())
        .fillna(0.0)
    )
    max_recent = max(float(df["recent_conflicts"].max()), 1.0)
    df["recent_conflicts"] = (df["recent_conflicts"] / max_recent).clip(0.0, 1.0)
    return df.drop(columns=["_pressure"], errors="ignore")
