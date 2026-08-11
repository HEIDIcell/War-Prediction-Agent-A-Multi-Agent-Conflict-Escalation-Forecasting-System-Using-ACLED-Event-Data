"""Optional GDELT DOC enrichment for the War Prediction Agent.

This module adds a lightweight RAG/news layer without making the whole project
unreproducible. It queries the public GDELT DOC 2.0 API for recent or monthly
US-target news articles, caches article metadata locally, and merges simple news
features back into market_cases_sample.csv.

The script is optional: if internet access fails or GDELT rate-limits requests,
the ACLED/World Bank experiments still run. In that case, the script writes zero
GDELT features and still produces market_cases_with_gdelt.csv.

Recommended usage after building ACLED/World Bank market cases:

    python -m src.data.build_gdelt_rag_cache --cases data/processed/market_cases_sample.csv --months-last 18
    copy data\\processed\\market_cases_with_gdelt.csv data\\processed\\market_cases_sample.csv
    python -m src.experiments.run_architecture_comparison

Useful safer test run:

    python -m src.data.build_gdelt_rag_cache --months-last 3 --max-records 5 --sleep 10

Outputs:
    data/processed/gdelt_rag_cache.csv
    data/processed/market_cases_with_gdelt.csv
"""
from __future__ import annotations

import argparse
import hashlib
import random
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests


TARGET_NAMES = {
    "US-China": "China",
    "US-Russia": "Russia",
    "US-Iran": "Iran",
    "US-NorthKorea": "North Korea",
}

NEGATIVE_WORDS = [
    "war",
    "conflict",
    "attack",
    "missile",
    "strike",
    "sanction",
    "military",
    "troops",
    "clash",
    "tension",
    "threat",
    "crisis",
    "escalation",
    "invasion",
    "airstrike",
    "drone",
    "bombing",
    "retaliation",
    "hostility",
    "nuclear",
]

POSITIVE_WORDS = [
    "talks",
    "diplomacy",
    "meeting",
    "agreement",
    "cooperation",
    "peace",
    "summit",
    "negotiation",
    "dialogue",
    "ceasefire",
    "deal",
    "de-escalation",
    "deescalation",
    "truce",
]

CACHE_COLUMNS = [
    "dyad",
    "month",
    "title",
    "url",
    "domain",
    "sourcecountry",
    "language",
    "seendate",
    "tone_proxy",
    "cache_id",
]

MERGED_GDELT_COLUMNS = [
    "gdelt_article_count",
    "gdelt_avg_tone",
    "gdelt_negative_share",
]


def _safe_text(x) -> str:
    """Convert missing/non-string values into a safe string."""
    return x if isinstance(x, str) else ""


def lexical_tone(title: str, snippet: str = "") -> float:
    """Return a simple [-1, 1] sentiment proxy for GDELT article metadata.

    This is not a true sentiment model. It is a lightweight, reproducible lexical
    proxy used for the optional RAG/news enrichment layer.
    """
    text = (_safe_text(title) + " " + _safe_text(snippet)).lower()

    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    pos = sum(1 for w in POSITIVE_WORDS if w in text)

    if pos + neg == 0:
        return 0.0

    return float(np.clip((pos - neg) / (pos + neg), -1.0, 1.0))


def _standardise_month_value(x) -> str:
    """Convert month values into YYYY-MM format.

    Handles values such as:
    - 2024-01
    - 2024-01-01
    - Timestamp values
    """
    dt = pd.to_datetime(str(x) + "-01" if len(str(x)) == 7 else str(x), errors="coerce")
    if pd.isna(dt):
        return str(x)[:7]
    return dt.strftime("%Y-%m")


def _ensure_cache_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the GDELT cache always has the expected columns."""
    for col in CACHE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    df = df[CACHE_COLUMNS].copy()

    if "tone_proxy" in df.columns:
        df["tone_proxy"] = pd.to_numeric(df["tone_proxy"], errors="coerce").fillna(0.0)

    return df


def _empty_cache_df() -> pd.DataFrame:
    """Return an empty GDELT cache table with valid columns."""
    return pd.DataFrame(columns=CACHE_COLUMNS)


def gdelt_doc_query(
    dyad: str,
    month: str,
    max_records: int = 5,
    timeout: int = 45,
    retries: int = 3,
    sleep_after_success: float = 8.0,
) -> List[Dict]:
    """Query GDELT DOC 2.0 for a dyad-month and return article metadata.

    This function is intentionally defensive:
    - handles 429 rate limits;
    - handles empty or non-JSON responses;
    - retries network failures;
    - returns [] instead of crashing.

    The public DOC API does not require registration. It returns article lists
    rather than a perfect social-science event label, so the results are used
    only for the RAG/explanation layer and weak news-sentiment features.
    """
    dyad = str(dyad)
    month = _standardise_month_value(month)

    target = TARGET_NAMES.get(dyad, dyad.replace("US-", "").replace("-", " "))

    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthEnd(0)

    start_s = start.strftime("%Y%m%d000000")
    end_s = end.strftime("%Y%m%d235959")

    query = (
        f'("United States" OR "U.S." OR "US") "{target}" '
        f"(military OR conflict OR diplomacy OR sanction OR security OR war OR tension)"
    )

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "sort": "hybridrel",
        "maxrecords": int(max_records),
        "startdatetime": start_s,
        "enddatetime": end_s,
    }

    headers = {
        "User-Agent": "war-prediction-coursework/1.0"
    }

    articles: List[Dict] = []

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout, headers=headers)

            if response.status_code == 429:
                wait_seconds = 45 + attempt * 45 + random.uniform(0, 10)
                print(
                    f"[WARNING] GDELT rate limit for {dyad} {month}. "
                    f"Waiting {wait_seconds:.1f}s before retry {attempt + 1}/{retries}."
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            text = response.text.strip()
            if not text:
                print(f"[WARNING] Empty GDELT response for {dyad} {month}.")
                break

            try:
                payload = response.json()
            except ValueError:
                print(f"[WARNING] Non-JSON GDELT response for {dyad} {month}. Skipping.")
                break

            if isinstance(payload, dict):
                articles = payload.get("articles", []) or []
            else:
                articles = []

            time.sleep(float(sleep_after_success) + random.uniform(0, 2))
            break

        except requests.exceptions.RequestException as exc:
            wait_seconds = 20 + attempt * 20 + random.uniform(0, 5)
            print(
                f"[WARNING] GDELT request failed for {dyad} {month}: {exc}. "
                f"Retrying in {wait_seconds:.1f}s."
            )
            time.sleep(wait_seconds)

    rows: List[Dict] = []

    for art in articles:
        if not isinstance(art, dict):
            continue

        title = _safe_text(art.get("title"))
        url_value = _safe_text(art.get("url"))
        domain = _safe_text(art.get("domain"))
        sourcecountry = _safe_text(art.get("sourcecountry"))
        language = _safe_text(art.get("language"))
        seendate = _safe_text(art.get("seendate"))

        snippet = " ".join([title, domain, sourcecountry, language, seendate])

        cache_id = hashlib.md5(
            (dyad + month + url_value + title).encode("utf-8", errors="ignore")
        ).hexdigest()

        rows.append(
            {
                "dyad": dyad,
                "month": month,
                "title": title,
                "url": url_value,
                "domain": domain,
                "sourcecountry": sourcecountry,
                "language": language,
                "seendate": seendate,
                "tone_proxy": lexical_tone(title, snippet),
                "cache_id": cache_id,
            }
        )

    return rows


def load_existing_cache(cache_path: str) -> pd.DataFrame:
    """Load existing GDELT cache safely."""
    path = Path(cache_path)

    if not path.exists():
        return _empty_cache_df()

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[WARNING] Existing cache {cache_path} is empty. Rebuilding with valid columns.")
        return _empty_cache_df()

    return _ensure_cache_columns(df)


def build_cache(
    cases_path: str,
    out_path: str,
    months_last: int = 18,
    max_records: int = 5,
    sleep: float = 8.0,
    timeout: int = 45,
    retries: int = 3,
    append_existing: bool = True,
) -> pd.DataFrame:
    """Build or update local GDELT article cache."""
    cases = pd.read_csv(cases_path)

    if "dyad" not in cases.columns or "month" not in cases.columns:
        raise ValueError("The cases file must contain 'dyad' and 'month' columns.")

    cases = cases.copy()
    cases["month"] = cases["month"].apply(_standardise_month_value)
    cases["month_dt"] = pd.to_datetime(cases["month"] + "-01", errors="coerce")
    cases = cases.dropna(subset=["month_dt"])

    if months_last and months_last > 0:
        cutoff = cases["month_dt"].max() - pd.DateOffset(months=months_last - 1)
        cases = cases[cases["month_dt"] >= cutoff]

    pairs = (
        cases[["dyad", "month"]]
        .drop_duplicates()
        .sort_values(["dyad", "month"])
        .reset_index(drop=True)
    )

    existing = load_existing_cache(out_path) if append_existing else _empty_cache_df()
    existing_keys = set()

    if not existing.empty:
        existing["month"] = existing["month"].apply(_standardise_month_value)
        existing_keys = set(zip(existing["dyad"].astype(str), existing["month"].astype(str)))

    new_rows: List[Dict] = []

    print(f"[INFO] GDELT dyad-month queries to consider: {len(pairs)}")
    print(f"[INFO] Existing cached dyad-months: {len(existing_keys)}")

    for _, r in pairs.iterrows():
        dyad = str(r["dyad"])
        month = _standardise_month_value(r["month"])

        if (dyad, month) in existing_keys:
            print(f"[INFO] {dyad} {month}: already cached, skipping")
            continue

        fetched = gdelt_doc_query(
            dyad=dyad,
            month=month,
            max_records=max_records,
            timeout=timeout,
            retries=retries,
            sleep_after_success=sleep,
        )

        new_rows.extend(fetched)

        print(f"[INFO] {dyad} {month}: {len(fetched)} GDELT articles")

        # Extra delay between dyad-month requests, including failed requests.
        time.sleep(float(sleep) + random.uniform(0, 2))

    new_df = pd.DataFrame(new_rows)
    new_df = _ensure_cache_columns(new_df)

    if append_existing:
        cache_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        cache_df = new_df

    cache_df = _ensure_cache_columns(cache_df)

    if not cache_df.empty:
        cache_df = cache_df.drop_duplicates(subset=["cache_id"], keep="first")
        cache_df = cache_df.sort_values(["dyad", "month", "seendate", "title"], na_position="last")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Wrote {out_path}: {cache_df.shape}")

    return cache_df


def merge_features(cases_path: str, cache_path: str, out_path: str) -> pd.DataFrame:
    """Merge GDELT article-level cache into dyad-month feature table."""
    cases = pd.read_csv(cases_path)

    if "dyad" not in cases.columns or "month" not in cases.columns:
        raise ValueError("The cases file must contain 'dyad' and 'month' columns.")

    cases = cases.copy()
    cases["month"] = cases["month"].apply(_standardise_month_value)

    cache = load_existing_cache(cache_path)

    if cache.empty:
        print(
            f"[WARNING] GDELT cache has no article rows. "
            f"Writing cases with zero GDELT features."
        )
        cases["gdelt_article_count"] = 0
        cases["gdelt_avg_tone"] = 0.0
        cases["gdelt_negative_share"] = 0.0

    else:
        cache = cache.copy()
        cache["month"] = cache["month"].apply(_standardise_month_value)
        cache["tone_proxy"] = pd.to_numeric(cache["tone_proxy"], errors="coerce").fillna(0.0)
        cache["is_negative"] = (cache["tone_proxy"] < 0).astype(int)

        agg = (
            cache.groupby(["dyad", "month"])
            .agg(
                gdelt_article_count=("cache_id", "count"),
                gdelt_avg_tone=("tone_proxy", "mean"),
                gdelt_negative_share=("is_negative", "mean"),
            )
            .reset_index()
        )

        cases = cases.merge(agg, on=["dyad", "month"], how="left")

        for col in MERGED_GDELT_COLUMNS:
            if col not in cases.columns:
                cases[col] = 0.0

        cases["gdelt_article_count"] = pd.to_numeric(
            cases["gdelt_article_count"], errors="coerce"
        ).fillna(0).astype(int)

        cases["gdelt_avg_tone"] = pd.to_numeric(
            cases["gdelt_avg_tone"], errors="coerce"
        ).fillna(0.0)

        cases["gdelt_negative_share"] = pd.to_numeric(
            cases["gdelt_negative_share"], errors="coerce"
        ).fillna(0.0)

        # Blend existing news_sentiment with GDELT lexical tone where available.
        # This keeps the original ACLED/World Bank feature table compatible while
        # allowing GDELT to influence the news sentiment signal.
        if "news_sentiment" in cases.columns:
            cases["news_sentiment"] = pd.to_numeric(
                cases["news_sentiment"], errors="coerce"
            ).fillna(0.0)

            cases["news_sentiment"] = np.where(
                cases["gdelt_article_count"] > 0,
                0.6 * cases["news_sentiment"] + 0.4 * cases["gdelt_avg_tone"],
                cases["news_sentiment"],
            )

    for col in MERGED_GDELT_COLUMNS:
        if col not in cases.columns:
            cases[col] = 0.0

    cases["gdelt_article_count"] = pd.to_numeric(
        cases["gdelt_article_count"], errors="coerce"
    ).fillna(0).astype(int)

    cases["gdelt_avg_tone"] = pd.to_numeric(
        cases["gdelt_avg_tone"], errors="coerce"
    ).fillna(0.0)

    cases["gdelt_negative_share"] = pd.to_numeric(
        cases["gdelt_negative_share"], errors="coerce"
    ).fillna(0.0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Wrote {out_path}: {cases.shape}")
    print("[INFO] GDELT feature summary:")
    print(cases[MERGED_GDELT_COLUMNS].describe())

    return cases


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        default="data/processed/market_cases_sample.csv",
        help="Input dyad-month cases CSV.",
    )

    parser.add_argument(
        "--cache",
        default="data/processed/gdelt_rag_cache.csv",
        help="Output/input local GDELT article cache CSV.",
    )

    parser.add_argument(
        "--out",
        default="data/processed/market_cases_with_gdelt.csv",
        help="Output dyad-month cases CSV with merged GDELT features.",
    )

    parser.add_argument(
        "--months-last",
        type=int,
        default=18,
        help="Only query the most recent N months in the cases file. Use 0 for all months.",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=5,
        help="Maximum GDELT articles per dyad-month query. Lower values reduce rate-limit risk.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=8.0,
        help="Seconds to sleep between GDELT requests. Use 8-15 if rate-limited.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="HTTP request timeout in seconds.",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per dyad-month query.",
    )

    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip API queries and only merge existing cache into cases.",
    )

    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Do not append to an existing cache. Rebuild cache from scratch.",
    )

    args = parser.parse_args()

    if not args.merge_only:
        build_cache(
            cases_path=args.cases,
            out_path=args.cache,
            months_last=args.months_last,
            max_records=args.max_records,
            sleep=args.sleep,
            timeout=args.timeout,
            retries=args.retries,
            append_existing=not args.no_append,
        )

    merge_features(
        cases_path=args.cases,
        cache_path=args.cache,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()