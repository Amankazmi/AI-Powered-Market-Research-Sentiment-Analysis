from __future__ import annotations

from typing import Iterable
import pandas as pd
import yake


def extract_keywords(df: pd.DataFrame, label_filter: str | None = None, text_column: str = "text",
                     max_ngram_size: int = 2, num_keywords: int = 20, language: str = "en") -> pd.DataFrame:
    """Extract keywords from the corpus using YAKE.

    Optionally filter by sentiment label (e.g., "Negative") before extraction.
    Returns a DataFrame with columns: keyword, score
    """
    corpus_df = df
    if label_filter and "sentiment_label" in df.columns:
        corpus_df = df[df["sentiment_label"] == label_filter]

    texts: Iterable[str] = corpus_df[text_column].dropna().astype(str).tolist() if text_column in corpus_df.columns else []
    joined = "\n".join(texts)

    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=num_keywords)
    keywords = kw_extractor.extract_keywords(joined)
    # YAKE returns (keyword, score) where lower score is better
    result = pd.DataFrame(keywords, columns=["keyword", "score"]).sort_values("score")
    return result
