
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.data.load_cases import FEATURE_COLUMNS

class CaseRetriever:
    def __init__(self, k=3):
        self.k = k
        self.train_df = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.train_matrix = None

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
