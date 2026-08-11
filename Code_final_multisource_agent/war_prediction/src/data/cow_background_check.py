"""Optional COW NMC/MID background coverage checker.

COW NMC and MID are not merged into the 2016-2025 ACLED training labels because
most relevant COW files end before the recent ACLED period used here. This
script creates a small report if the user places COW CSV files in data/raw/cow/.
It is intended for background/historical comparison and report transparency.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def inspect_file(path: Path) -> dict:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return {"file": path.name, "rows": 0, "min_year": None, "max_year": None, "note": "could not read"}
    year_col = next((c for c in df.columns if c.lower() in ["year", "styear", "endyear"]), None)
    if year_col:
        years = pd.to_numeric(df[year_col], errors="coerce")
        min_year = int(years.min()) if years.notna().any() else None
        max_year = int(years.max()) if years.notna().any() else None
    else:
        min_year = max_year = None
    return {"file": path.name, "rows": len(df), "min_year": min_year, "max_year": max_year, "note": "background only"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/cow")
    ap.add_argument("--out", default="results/tables/cow_background_coverage.csv")
    args = ap.parse_args()
    paths = list(Path(args.raw_dir).glob("*.csv"))
    rows = [inspect_file(p) for p in paths]
    if not rows:
        rows = [{"file": "NO_COW_FILES_FOUND", "rows": 0, "min_year": None, "max_year": None, "note": "COW is documented as background/future work"}]
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out)
    print(f"[INFO] Wrote {args.out}")


if __name__ == "__main__":
    main()
