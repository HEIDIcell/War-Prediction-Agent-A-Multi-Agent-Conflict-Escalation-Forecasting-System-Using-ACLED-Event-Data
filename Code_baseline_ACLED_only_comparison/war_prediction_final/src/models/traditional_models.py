
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.data.load_cases import FEATURE_COLUMNS

def make_logistic_regression():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

def make_random_forest():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42,
        )),
    ])

def fit_predict_model(model, train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_COLUMNS]
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return np.clip(proba, 0.001, 0.999)
