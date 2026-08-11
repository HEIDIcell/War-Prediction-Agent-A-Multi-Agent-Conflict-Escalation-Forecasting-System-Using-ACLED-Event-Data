"""Build real market cases from public GDELT 1.0 event files and COW Dyadic MID labels.

This script integrates the data-acquisition step into the project.

Default scope:
    2000-2012 inclusive (13 years), four US-target dyads:
    US-China, US-Russia, US-Iran, US-NorthKorea.

Output:
    data/processed/market_cases_sample.csv

Important design choice:
    Experiments run on a fixed local CSV after this script has downloaded and
    processed public data. This is more reproducible than calling a live API
    during each experiment run.

Data notes:
    - GDELT 1.0 is used because it overlaps with COW MID labels.
    - GDELT 1979-2005 event data is distributed as yearly ZIP files.
    - GDELT 2006-March 2013 event data is distributed as monthly ZIP files.
    - COW Dyadic MID is used to label whether a dyad experiences a militarized
      interstate dispute within the future prediction horizon.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GDELT_RAW_DIR = RAW_DIR / "gdelt"
COW_RAW_DIR = RAW_DIR / "cow_mid"

GDELT_BASE = "https://data.gdeltproject.org/events"
DYADIC_MID_URL = "https://correlatesofwar.org/wp-content/uploads/dyadic_mid_4.03_update.zip"

# GDELT CAMEO actor country codes and COW state codes.
DYADS = {
    "US-China": {"gdelt": ("USA", "CHN"), "cow": (2, 710)},
    "US-Russia": {"gdelt": ("USA", "RUS"), "cow": (2, 365)},
    "US-Iran": {"gdelt": ("USA", "IRN"), "cow": (2, 630)},
    "US-NorthKorea": {"gdelt": ("USA", "PRK"), "cow": (2, 731)},
}

# Selected GDELT 1.0 Event Database columns.
GDELT_USECOLS = [
    1,   # SQLDATE
    7,   # Actor1CountryCode
    17,  # Actor2CountryCode
    26,  # EventCode
    27,  # EventBaseCode
    28,  # EventRootCode
    29,  # QuadClass
    30,  # GoldsteinScale
    34,  # AvgTone
]
GDELT_NAMES = [
    "sql_date",
    "actor1_country",
    "actor2_country",
    "event_code",
    "event_base_code",
    "event_root_code",
    "quad_class",
    "goldstein",
    "avg_tone",
]
FEATURE_COLUMNS = [
    "conflict_count", "military_count", "threat_count",
    "cooperation_count", "diplomatic_count", "avg_tone", "avg_goldstein",
]


def month_range(start_year: int, end_year: int) -> List[str]:
    return [f"{y}-{m:02d}" for y in range(start_year, end_year + 1) for m in range(1, 13)]


def download_file(url: str, out_path: Path, timeout: int = 90) -> bool:
    """Download URL to out_path unless already cached."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[CACHE] {out_path}")
        return True
    try:
        print(f"[DOWNLOAD] {url}")
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 coursework-data-downloader"})
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
        out_path.write_bytes(data)
        print(f"[OK] {out_path.name} ({len(data)/1e6:.1f} MB)")
        return True
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"[WARN] Could not download {url}: {exc}")
        return False


def gdelt_file_candidates_for_year(year: int) -> List[Tuple[str, str]]:
    """Candidate yearly GDELT 1.0 event filenames for 1979-2005."""
    return [
        (f"{GDELT_BASE}/{year}.zip", f"{year}.zip"),
        (f"{GDELT_BASE}/{year}.export.CSV.zip", f"{year}.export.CSV.zip"),
    ]


def gdelt_file_candidates_for_month(year: int, month: int) -> List[Tuple[str, str]]:
    """Candidate monthly GDELT 1.0 event filenames for 2006-March 2013."""
    ym = f"{year}{month:02d}"
    return [
        (f"{GDELT_BASE}/{ym}.zip", f"{ym}.zip"),
        (f"{GDELT_BASE}/{ym}.export.CSV.zip", f"{ym}.export.CSV.zip"),
    ]


def ensure_first_available(candidates: List[Tuple[str, str]]) -> Optional[Path]:
    for url, filename in candidates:
        out_path = GDELT_RAW_DIR / filename
        if download_file(url, out_path):
            return out_path
    return None


def init_accumulators(months: Iterable[str]) -> Dict[Tuple[str, str], Dict[str, float]]:
    acc: Dict[Tuple[str, str], Dict[str, float]] = {}
    for dyad in DYADS:
        for month in months:
            acc[(dyad, month)] = {
                "conflict_count": 0.0,
                "military_count": 0.0,
                "threat_count": 0.0,
                "cooperation_count": 0.0,
                "diplomatic_count": 0.0,
                "tone_sum": 0.0,
                "tone_n": 0.0,
                "goldstein_sum": 0.0,
                "goldstein_n": 0.0,
            }
    return acc


def update_accumulators_from_chunk(chunk: pd.DataFrame, acc: Dict[Tuple[str, str], Dict[str, float]], allowed_months: set[str]) -> None:
    """Filter one GDELT chunk to target dyads and update monthly aggregate accumulators."""
    chunk = chunk.copy()
    chunk["month"] = pd.to_datetime(chunk["sql_date"].astype(str), format="%Y%m%d", errors="coerce").dt.to_period("M").astype(str)
    chunk = chunk[chunk["month"].isin(allowed_months)]
    if chunk.empty:
        return

    chunk["quad_class"] = pd.to_numeric(chunk["quad_class"], errors="coerce").fillna(0).astype(int)
    chunk["goldstein"] = pd.to_numeric(chunk["goldstein"], errors="coerce")
    chunk["avg_tone"] = pd.to_numeric(chunk["avg_tone"], errors="coerce")
    for col in ["event_code", "event_base_code", "event_root_code"]:
        chunk[col] = chunk[col].astype("string").str.zfill(2)

    for dyad_name, spec in DYADS.items():
        a, b = spec["gdelt"]
        mask = ((chunk["actor1_country"] == a) & (chunk["actor2_country"] == b)) | ((chunk["actor1_country"] == b) & (chunk["actor2_country"] == a))
        sub = chunk.loc[mask].copy()
        if sub.empty:
            continue

        root = sub["event_root_code"].astype(str).str[:2]
        base = sub["event_base_code"].astype(str).str[:2]
        sub["conflict_count"] = (sub["quad_class"].eq(4) | root.isin(["18", "19", "20"])).astype(int)
        sub["military_count"] = root.isin(["15", "18", "19", "20"]).astype(int)
        sub["threat_count"] = (root.eq("13") | base.eq("13")).astype(int)
        sub["cooperation_count"] = (sub["quad_class"].isin([1, 2]) | root.isin(["01", "02", "03", "04", "05", "06", "07", "08"])).astype(int)
        sub["diplomatic_count"] = root.isin(["03", "04", "05"]).astype(int)
        sub["tone_valid"] = sub["avg_tone"].notna().astype(int)
        sub["goldstein_valid"] = sub["goldstein"].notna().astype(int)
        sub["avg_tone_filled"] = sub["avg_tone"].fillna(0.0)
        sub["goldstein_filled"] = sub["goldstein"].fillna(0.0)

        grouped = sub.groupby("month", as_index=True).agg({
            "conflict_count": "sum",
            "military_count": "sum",
            "threat_count": "sum",
            "cooperation_count": "sum",
            "diplomatic_count": "sum",
            "avg_tone_filled": "sum",
            "tone_valid": "sum",
            "goldstein_filled": "sum",
            "goldstein_valid": "sum",
        })
        for month, row in grouped.iterrows():
            key = (dyad_name, str(month))
            if key not in acc:
                continue
            target = acc[key]
            for c in ["conflict_count", "military_count", "threat_count", "cooperation_count", "diplomatic_count"]:
                target[c] += float(row[c])
            target["tone_sum"] += float(row["avg_tone_filled"])
            target["tone_n"] += float(row["tone_valid"])
            target["goldstein_sum"] += float(row["goldstein_filled"])
            target["goldstein_n"] += float(row["goldstein_valid"])


def process_gdelt_zip(zip_path: Path, acc: Dict[Tuple[str, str], Dict[str, float]], allowed_months: Iterable[str], chunksize: int = 250_000) -> None:
    allowed = set(allowed_months)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt", ".tsv"))]
        if not names:
            names = zf.namelist()
        inner = names[0]
        print(f"[PROCESS] {zip_path.name} -> {inner}")
        with zf.open(inner) as f:
            reader = pd.read_csv(
                f,
                sep="\t",
                header=None,
                usecols=GDELT_USECOLS,
                names=GDELT_NAMES,
                dtype={
                    "actor1_country": "string",
                    "actor2_country": "string",
                    "event_code": "string",
                    "event_base_code": "string",
                    "event_root_code": "string",
                },
                chunksize=chunksize,
                low_memory=False,
            )
            for i, chunk in enumerate(reader, start=1):
                update_accumulators_from_chunk(chunk, acc, allowed)
                if i % 20 == 0:
                    print(f"  processed {i * chunksize:,} rows from {zip_path.name}...")


def accumulators_to_features(acc: Dict[Tuple[str, str], Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for (dyad, month), vals in sorted(acc.items(), key=lambda x: (x[0][0], x[0][1])):
        rows.append({
            "dyad": dyad,
            "month": month,
            "conflict_count": int(vals["conflict_count"]),
            "military_count": int(vals["military_count"]),
            "threat_count": int(vals["threat_count"]),
            "cooperation_count": int(vals["cooperation_count"]),
            "diplomatic_count": int(vals["diplomatic_count"]),
            "avg_tone": float(vals["tone_sum"] / vals["tone_n"]) if vals["tone_n"] > 0 else 0.0,
            "avg_goldstein": float(vals["goldstein_sum"] / vals["goldstein_n"]) if vals["goldstein_n"] > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def build_gdelt_features(start_year: int, end_year: int) -> pd.DataFrame:
    months = month_range(start_year, end_year)
    acc = init_accumulators(months)

    for year in range(start_year, end_year + 1):
        if year <= 2005:
            year_months = [f"{year}-{m:02d}" for m in range(1, 13)]
            zip_path = ensure_first_available(gdelt_file_candidates_for_year(year))
            if zip_path is None:
                print(f"[WARN] No yearly GDELT file found for {year}; zeros will remain for that year.")
                continue
            process_gdelt_zip(zip_path, acc, year_months)
        else:
            for month in range(1, 13):
                month_str = f"{year}-{month:02d}"
                zip_path = ensure_first_available(gdelt_file_candidates_for_month(year, month))
                if zip_path is None:
                    print(f"[WARN] No monthly GDELT file found for {month_str}; zeros will remain for that month.")
                    continue
                process_gdelt_zip(zip_path, acc, [month_str])

    return accumulators_to_features(acc)


def download_and_read_dyadic_mid() -> pd.DataFrame:
    COW_RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = COW_RAW_DIR / "dyadic_mid_4.03_update.zip"
    if not download_file(DYADIC_MID_URL, zip_path):
        raise RuntimeError(
            "Could not download Dyadic MID automatically. Manually download dyadic_mid_4.03_update.zip "
            "from the Correlates of War MID page and place it at data/raw/cow_mid/dyadic_mid_4.03_update.zip"
        )
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV file found in {zip_path}")
        chosen = sorted(csv_names, key=lambda n: ("codebook" in n.lower(), len(n)))[0]
        print(f"[PROCESS] COW Dyadic MID file: {chosen}")
        with zf.open(chosen) as f:
            mid = pd.read_csv(f, low_memory=False)
    mid.columns = [str(c).strip() for c in mid.columns]
    return mid


def find_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    norm_map = {c.lower().replace("_", "").replace(" ", ""): c for c in cols}
    for cand in candidates:
        key = cand.lower().replace("_", "").replace(" ", "")
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = cand.lower().replace("_", "").replace(" ", "")
        for norm, original in norm_map.items():
            if key in norm:
                return original
    return None


def mid_date(row: pd.Series, y_col: str, m_col: Optional[str], d_col: Optional[str]) -> pd.Timestamp:
    y = int(row[y_col])
    m = int(row[m_col]) if m_col and not pd.isna(row[m_col]) else 1
    d = int(row[d_col]) if d_col and not pd.isna(row[d_col]) else 1
    m = min(max(m, 1), 12)
    d = min(max(d, 1), 28)
    return pd.Timestamp(year=y, month=m, day=d)


def build_mid_intervals(mid: pd.DataFrame) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    cols = mid.columns
    c1 = find_col(cols, ["ccode1", "state1", "stnum1", "sidea", "statea", "cowcode1"])
    c2 = find_col(cols, ["ccode2", "state2", "stnum2", "sideb", "stateb", "cowcode2"])
    a1 = find_col(cols, ["stabb1", "stateabb1", "stateab1"])
    a2 = find_col(cols, ["stabb2", "stateabb2", "stateab2"])

    sy = find_col(cols, ["strtyr", "startyear", "styear", "year", "startyr"])
    sm = find_col(cols, ["strtmnth", "startmonth", "stmonth", "startmon", "strtmon"])
    sd = find_col(cols, ["strtday", "startday", "stday"])
    ey = find_col(cols, ["endyear", "endyr", "endyr1"])
    em = find_col(cols, ["endmnth", "endmonth", "endmon"])
    ed = find_col(cols, ["endday"])

    if sy is None:
        raise RuntimeError(f"Could not find MID start year column. Available columns: {list(cols)}")

    intervals: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for _, row in mid.iterrows():
        row_dyad = None
        for dyad_name, spec in DYADS.items():
            cow_a, cow_b = spec["cow"]
            gdelt_a, gdelt_b = spec["gdelt"]
            matched = False
            if c1 and c2:
                try:
                    x, y = int(row[c1]), int(row[c2])
                    matched = {x, y} == {cow_a, cow_b}
                except Exception:
                    matched = False
            if not matched and a1 and a2:
                x, y = str(row[a1]).upper(), str(row[a2]).upper()
                matched = {x, y} == {gdelt_a, gdelt_b}
            if matched:
                row_dyad = dyad_name
                break
        if row_dyad is None:
            continue
        start = mid_date(row, sy, sm, sd)
        end = mid_date(row, ey, em, ed) if ey else start
        if end < start:
            end = start
        intervals.append((row_dyad, start, end))
    print(f"[INFO] Matched {len(intervals)} dyadic MID intervals for selected US-target dyads.")
    return intervals


def label_cases(features: pd.DataFrame, intervals: List[Tuple[str, pd.Timestamp, pd.Timestamp]], horizon_months: int) -> pd.DataFrame:
    labels: List[int] = []
    for _, row in features.iterrows():
        dyad = row["dyad"]
        current_month = pd.Period(row["month"], freq="M")
        start_window = (current_month + 1).to_timestamp(how="start")
        end_window = (current_month + horizon_months).to_timestamp(how="end")
        label = 0
        for mid_dyad, mid_start, mid_end in intervals:
            if mid_dyad != dyad:
                continue
            if mid_start <= end_window and mid_end >= start_window:
                label = 1
                break
        labels.append(label)
    out = features.copy()
    out["label"] = labels
    return out


def add_model_probability(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].fillna(0).astype(float)
    y = df["label"].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 3:
        print("[WARN] Too few labels for cross-validated logistic model. Using heuristic model_probability.")
        raw = (
            0.10 * X["conflict_count"] + 0.16 * X["military_count"] + 0.14 * X["threat_count"]
            - 0.04 * X["cooperation_count"] - 0.03 * X["diplomatic_count"]
            - 0.22 * X["avg_tone"] - 0.16 * X["avg_goldstein"]
        )
        prob = 1 / (1 + np.exp(-(raw - raw.mean()) / (raw.std() + 1e-6)))
    else:
        n_splits = min(5, int(y.value_counts().min()))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    out = df.copy()
    out["model_probability"] = np.clip(prob, 0.01, 0.99)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 13-year GDELT+COW market cases for the architecture experiment.")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2012)
    parser.add_argument("--horizon-months", type=int, default=6)
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR / "market_cases_sample.csv"))
    args = parser.parse_args()

    if args.end_year < args.start_year:
        raise ValueError("end-year must be >= start-year")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building GDELT features for {args.start_year}-{args.end_year} inclusive.")
    print(f"[INFO] This is {args.end_year - args.start_year + 1} years and {len(month_range(args.start_year, args.end_year))} months.")
    features = build_gdelt_features(args.start_year, args.end_year)
    features_out = PROCESSED_DIR / "gdelt_monthly_features.csv"
    features.to_csv(features_out, index=False)
    print(f"[OK] Wrote intermediate features: {features_out} ({len(features)} rows)")

    print("[INFO] Building COW MID labels...")
    mid = download_and_read_dyadic_mid()
    intervals = build_mid_intervals(mid)
    cases = label_cases(features, intervals, args.horizon_months)
    cases = add_model_probability(cases)
    cases["case_id"] = cases["dyad"] + "_" + cases["month"]
    cases = cases[[
        "case_id", "dyad", "month",
        "conflict_count", "military_count", "threat_count",
        "cooperation_count", "diplomatic_count",
        "avg_tone", "avg_goldstein", "model_probability", "label",
    ]]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out, index=False)

    summary = cases.groupby("dyad")["label"].agg(["count", "sum", "mean"]).reset_index()
    summary_out = PROCESSED_DIR / "market_cases_summary.csv"
    summary.to_csv(summary_out, index=False)

    print(f"[OK] Wrote final market cases: {out} ({len(cases)} rows)")
    print(f"[OK] Wrote summary: {summary_out}")
    print("[INFO] Label distribution:")
    print(cases["label"].value_counts().sort_index())
    print("[INFO] Label summary by dyad:")
    print(summary.to_string(index=False))
    print(cases.head().to_string(index=False))


if __name__ == "__main__":
    main()
