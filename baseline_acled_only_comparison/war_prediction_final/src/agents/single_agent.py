
import time
import numpy as np
import pandas as pd

class SingleRAGAgent:
    def __init__(self, retriever, name="Single-Agent RAG Analyst"):
        self.retriever = retriever
        self.name = name

    def predict_one(self, row: pd.Series, model_probability: float):
        start = time.perf_counter()
        retrieved = self.retriever.retrieve(row)
        retrieved_rate = float(retrieved["label"].mean()) if len(retrieved) else 0.5
        p = float(np.clip(0.72 * float(model_probability) + 0.28 * retrieved_rate, 0.001, 0.999))
        top = retrieved.iloc[0] if len(retrieved) else None
        retrieved_text = (
            f"The most similar historical case was {top['dyad']} in {top['month']} with label={int(top['label'])}."
            if top is not None else "No similar historical case was retrieved."
        )
        direction = "high/elevated" if p >= 0.5 else "low/reduced"
        explanation = (
            f"The model probability is {p:.2f}, indicating {direction} conflict risk. "
            f"The agent considers conflict={row['conflict_count']}, military={row['military_count']}, "
            f"threat={row['threat_count']}, cooperation={row['cooperation_count']}, "
            f"diplomatic={row['diplomatic_count']}, tone={row['avg_tone']:.2f}, "
            f"and Goldstein={row['avg_goldstein']:.2f}. "
            f"Retrieved similar historical cases suggest a positive-case rate of {retrieved_rate:.2f}. "
            f"{retrieved_text}"
        )
        return p, explanation, time.perf_counter() - start

    def predict(self, test_df: pd.DataFrame, model_probabilities):
        probs, explanations, runtimes = [], [], []
        for (_, row), mp in zip(test_df.iterrows(), model_probabilities):
            p, e, rt = self.predict_one(row, float(mp))
            probs.append(p); explanations.append(e); runtimes.append(rt)
        return np.asarray(probs), explanations, np.asarray(runtimes)
