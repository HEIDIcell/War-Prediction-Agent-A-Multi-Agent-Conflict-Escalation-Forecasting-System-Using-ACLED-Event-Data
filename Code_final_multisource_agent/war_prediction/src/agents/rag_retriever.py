from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.data.load_cases import FEATURE_COLUMNS

class CaseRetriever:
    """Local RAG-style retriever for similar historical cases.

    The core retrieval is vector similarity over processed dyad-month features.
    If a GDELT DOC cache exists at data/processed/gdelt_rag_cache.csv, the
    retriever can also surface article titles for the same dyad-month in the
    generated explanation. This keeps the system reproducible while allowing an
    optional news/RAG layer.
    """

    def __init__(self, k: int = 3, gdelt_cache_path: str = "data/processed/gdelt_rag_cache.csv"):
        self.k = k
        self.train_df = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.train_matrix = None
        self.gdelt_cache_path = Path(gdelt_cache_path)
        self.gdelt_cache = None
        if self.gdelt_cache_path.exists():
            try:
                self.gdelt_cache = pd.read_csv(self.gdelt_cache_path)
            except Exception:
                self.gdelt_cache = None

    def fit(self, train_df: pd.DataFrame):
        self.train_df = train_df.reset_index(drop=True).copy()
        X = self.imputer.fit_transform(self.train_df[FEATURE_COLUMNS])
        self.train_matrix = self.scaler.fit_transform(X)
        return self

    def retrieve(self, row: pd.Series) -> pd.DataFrame:
        X = self.imputer.transform(pd.DataFrame([row[FEATURE_COLUMNS].to_dict()]))
        X = self.scaler.transform(X)
        sims = cosine_similarity(X, self.train_matrix)[0]
        top_idx = np.argsort(sims)[::-1][: self.k]
        result = self.train_df.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result

    def retrieve_articles(self, dyad: str, month: str, k: int = 3) -> list[str]:
        if self.gdelt_cache is None or self.gdelt_cache.empty:
            return []
        g = self.gdelt_cache[(self.gdelt_cache["dyad"] == dyad) & (self.gdelt_cache["month"] == month)].copy()
        if g.empty:
            return []
        if "tone_proxy" in g.columns:
            g["rank_abs_tone"] = g["tone_proxy"].abs()
            g = g.sort_values("rank_abs_tone", ascending=False)
        titles = []
        for _, r in g.head(k).iterrows():
            title = str(r.get("title", "")).strip()
            domain = str(r.get("domain", "")).strip()
            if title and title.lower() != "nan":
                titles.append(f"{title}" + (f" ({domain})" if domain and domain.lower() != "nan" else ""))
        return titles
