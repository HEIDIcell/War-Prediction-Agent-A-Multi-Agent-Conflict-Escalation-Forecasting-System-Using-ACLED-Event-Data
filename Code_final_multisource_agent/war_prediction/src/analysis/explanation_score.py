
import re

ESCALATION_TERMS = ["conflict", "military", "threat", "negative", "tone", "goldstein", "escalation"]
DEESCALATION_TERMS = ["cooperation", "diplomatic", "de-escalation", "deescalation", "positive"]
MODEL_TERMS = ["model", "probability", "risk", "forecast"]
RETRIEVAL_TERMS = ["similar", "historical", "retrieved", "case"]

def explanation_score(text: str, probability: float) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    score = 0.0
    if any(term in t for term in ESCALATION_TERMS): score += 0.25
    if any(term in t for term in DEESCALATION_TERMS): score += 0.20
    if any(term in t for term in MODEL_TERMS): score += 0.20
    if any(term in t for term in RETRIEVAL_TERMS): score += 0.15
    if re.search(r"\d+(\.\d+)?%|\b0\.\d+\b", t): score += 0.10
    if probability >= 0.5 and any(w in t for w in ["high", "elevated", "rising", "increase"]): score += 0.10
    elif probability < 0.5 and any(w in t for w in ["low", "reduced", "falling", "decrease"]): score += 0.10
    return min(score, 1.0)
