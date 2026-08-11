
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "case_id", "dyad", "month",
    "conflict_count", "military_count", "threat_count",
    "cooperation_count", "diplomatic_count",
    "avg_tone", "avg_goldstein",
    "model_probability", "label",
]

BASE_FEATURE_COLUMNS = [
    "conflict_count", "military_count", "threat_count",
    "cooperation_count", "diplomatic_count",
    "avg_tone", "avg_goldstein", "model_probability",
]

# GEO_CONTEXT features: optional but used automatically when present in the dataset.
CONTEXT_FEATURE_COLUMNS = [
    "cinc_ratio", "diplomatic_score", "trade_dependence",
    "news_sentiment", "recent_conflicts",
    "fatalities_sum", "us_actor_events",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS

def generate_synthetic_cases(path: Path, n_months: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dyads = ["US-China", "US-Russia", "US-Iran", "US-NorthKorea"]
    rows = []
    months = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    idx = 0
    for dyad in dyads:
        base_risk = {"US-China": 0.35, "US-Russia": 0.42, "US-Iran": 0.48, "US-NorthKorea": 0.45}[dyad]
        latent = base_risk
        for m in months:
            latent = np.clip(0.75 * latent + 0.25 * base_risk + rng.normal(0, 0.08), 0.02, 0.98)
            conflict = rng.poisson(4 + 18 * latent)
            military = rng.poisson(1 + 9 * latent)
            threat = rng.poisson(1 + 7 * latent)
            cooperation = rng.poisson(7 + 12 * (1 - latent))
            diplomacy = rng.poisson(4 + 8 * (1 - latent))
            tone = rng.normal(-6 * latent + 2 * (1 - latent), 1.5)
            goldstein = rng.normal(-5 * latent + 3 * (1 - latent), 1.2)
            raw_score = (
                0.06 * conflict + 0.09 * military + 0.08 * threat
                - 0.04 * cooperation - 0.04 * diplomacy
                - 0.08 * tone - 0.06 * goldstein
                + rng.normal(0, 0.6)
            )
            prob = 1 / (1 + np.exp(-(raw_score - 1.6)))
            label = int(rng.random() < prob)
            model_prob = np.clip(prob + rng.normal(0, 0.10), 0.01, 0.99)
            rows.append({
                "case_id": f"CASE_{idx:05d}",
                "dyad": dyad,
                "month": m.strftime("%Y-%m"),
                "conflict_count": conflict,
                "military_count": military,
                "threat_count": threat,
                "cooperation_count": cooperation,
                "diplomatic_count": diplomacy,
                "avg_tone": tone,
                "avg_goldstein": goldstein,
                "model_probability": model_prob,
                "label": label,
            })
            idx += 1
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

def load_market_cases(path="data/processed/market_cases_sample.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"[INFO] {path} not found. Creating synthetic demo data.")
        df = generate_synthetic_cases(path)
    else:
        df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    # Add optional context columns as zeros/defaults when the older demo dataset is used.
    for col in CONTEXT_FEATURE_COLUMNS:
        if col not in df.columns:
            if col == "cinc_ratio":
                df[col] = 1.0
            elif col == "trade_dependence":
                df[col] = 0.35
            else:
                df[col] = 0.0
    df["month_dt"] = pd.to_datetime(df["month"].astype(str) + "-01", errors="coerce")
    if df["month_dt"].isna().any():
        df["month_dt"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.sort_values(["dyad", "month_dt", "case_id"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df
