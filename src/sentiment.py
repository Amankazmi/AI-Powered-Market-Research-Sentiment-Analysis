from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Wraps VADER sentiment analysis and attaches labels/score.

    Labels: Positive / Neutral / Negative based on compound score thresholds.
    """

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05) -> None:
        self._analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def _label_from_compound(self, compound: float) -> str:
        if compound >= self.pos_threshold:
            return "Positive"
        if compound <= self.neg_threshold:
            return "Negative"
        return "Neutral"

    def score_text(self, text: str) -> dict[str, float | str]:
        scores = self._analyzer.polarity_scores(text or "")
        label = self._label_from_compound(scores.get("compound", 0.0))
        scores["label"] = label
        return scores

    def add_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            raise ValueError("DataFrame must include a 'text' column")
        out = df.copy()
        scores = out["text"].astype(str).apply(self.score_text)
        out["sentiment_compound"] = scores.apply(lambda d: d["compound"])  # type: ignore[index]
        out["sentiment_label"] = scores.apply(lambda d: d["label"])  # type: ignore[index]
        out["sentiment_pos"] = scores.apply(lambda d: d["pos"])  # type: ignore[index]
        out["sentiment_neu"] = scores.apply(lambda d: d["neu"])  # type: ignore[index]
        out["sentiment_neg"] = scores.apply(lambda d: d["neg"])  # type: ignore[index]
        return out
