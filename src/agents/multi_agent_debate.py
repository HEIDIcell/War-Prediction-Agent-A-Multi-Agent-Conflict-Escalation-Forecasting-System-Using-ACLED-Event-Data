
import time
import numpy as np
import pandas as pd

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

class SentimentAgent:
    def assess(self, row):
        score = -0.16 * float(row["avg_tone"]) - 0.12 * float(row["avg_goldstein"])
        p = _sigmoid(score)
        text = f"Sentiment Agent: tone={row['avg_tone']:.2f} and Goldstein={row['avg_goldstein']:.2f}; negative sentiment raises escalation risk."
        return float(np.clip(p, 0.001, 0.999)), text

class EscalationAgent:
    def assess(self, row):
        score = 0.08*float(row["conflict_count"]) + 0.13*float(row["military_count"]) + 0.14*float(row["threat_count"]) - 2.0
        p = _sigmoid(score)
        text = f"Escalation Agent: conflict={row['conflict_count']}, military={row['military_count']}, threat={row['threat_count']}; these conflict and military indicators support a rising risk view."
        return float(np.clip(p, 0.001, 0.999)), text

class DeescalationAgent:
    def assess(self, row):
        score = 0.09*float(row["cooperation_count"]) + 0.12*float(row["diplomatic_count"]) + 0.04*max(float(row["avg_goldstein"]), 0) - 1.5
        risk_p = 1.0 - _sigmoid(score)
        text = f"De-escalation Agent: cooperation={row['cooperation_count']} and diplomatic={row['diplomatic_count']}; cooperation and diplomatic signals reduce risk."
        return float(np.clip(risk_p, 0.001, 0.999)), text


class GeoContextAgent:
    def assess(self, row):
        # Higher recent_conflicts and negative news sentiment raise risk.
        # Higher diplomatic_score and trade_dependence reduce risk.
        score = (
            1.25 * float(row.get("recent_conflicts", 0.0))
            + 0.18 * float(row.get("cinc_ratio", 1.0))
            - 0.85 * float(row.get("diplomatic_score", 0.0))
            - 0.55 * float(row.get("trade_dependence", 0.35))
            - 0.95 * float(row.get("news_sentiment", 0.0))
            - 0.20
        )
        p = _sigmoid(score)
        text = (
            f"Geo-context Agent: cinc_ratio={float(row.get('cinc_ratio', 1.0)):.2f}, "
            f"diplomatic_score={float(row.get('diplomatic_score', 0.0)):.2f}, "
            f"trade_dependence={float(row.get('trade_dependence', 0.35)):.2f}, "
            f"news_sentiment={float(row.get('news_sentiment', 0.0)):.2f}, "
            f"recent_conflicts={float(row.get('recent_conflicts', 0.0)):.2f}. "
            f"Capability imbalance and recent conflict history are weighed against diplomacy, trade interdependence and news sentiment."
        )
        return float(np.clip(p, 0.001, 0.999)), text

class DataDrivenAgent:
    def assess(self, row, model_probability):
        p = float(np.clip(model_probability, 0.001, 0.999))
        return p, f"Data-driven Agent: the baseline model probability is {p:.2f}."

class JudgeAgent:
    def decide(self, row, agent_outputs, retrieved_rate, model_probability):
        probs = np.asarray([p for p, _ in agent_outputs], dtype=float)
        weights = np.asarray([0.15, 0.22, 0.16, 0.19, 0.28], dtype=float)
        p = float((probs * weights).sum() / weights.sum())
        p = float(np.clip(0.86 * p + 0.14 * retrieved_rate, 0.001, 0.999))
        direction = "high/elevated" if p >= 0.5 else "low/reduced"
        parts = " ".join(text for _, text in agent_outputs)
        explanation = (
            f"Judge Agent final probability is {p:.2f}, indicating {direction} risk. "
            f"{parts} Similar retrieved historical cases have positive-case rate {retrieved_rate:.2f}. "
            f"The debate considers both escalation evidence and de-escalation counter-evidence before the final forecast."
        )
        return p, explanation

class MultiAgentDebateSystem:
    def __init__(self, retriever):
        self.retriever = retriever
        self.sentiment = SentimentAgent()
        self.escalation = EscalationAgent()
        self.deescalation = DeescalationAgent()
        self.geo_context = GeoContextAgent()
        self.data = DataDrivenAgent()
        self.judge = JudgeAgent()

    def predict_one(self, row, model_probability):
        start = time.perf_counter()
        retrieved = self.retriever.retrieve(row)
        retrieved_rate = float(retrieved["label"].mean()) if len(retrieved) else 0.5
        outputs = [
            self.sentiment.assess(row),
            self.escalation.assess(row),
            self.deescalation.assess(row),
            self.geo_context.assess(row),
            self.data.assess(row, model_probability),
        ]
        p, explanation = self.judge.decide(row, outputs, retrieved_rate, model_probability)
        return p, explanation, time.perf_counter() - start

    def predict(self, test_df, model_probabilities):
        probs, explanations, runtimes = [], [], []
        for (_, row), mp in zip(test_df.iterrows(), model_probabilities):
            p, e, rt = self.predict_one(row, float(mp))
            probs.append(p); explanations.append(e); runtimes.append(rt)
        return np.asarray(probs), explanations, np.asarray(runtimes)
