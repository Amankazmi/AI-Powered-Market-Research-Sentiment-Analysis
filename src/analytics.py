from __future__ import annotations

import pandas as pd
import plotly.express as px


def compute_sentiment_over_time(df: pd.DataFrame):
    if "date" not in df.columns:
        return px.scatter(title="No date column found")
    tmp = df.dropna(subset=["date"]).copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.to_period("W").dt.start_time
    agg = tmp.groupby(["date", "sentiment_label"]).size().reset_index(name="count")
    fig = px.line(agg, x="date", y="count", color="sentiment_label", markers=True,
                  title="Sentiment counts over time")
    fig.update_layout(legend_title_text="Sentiment")
    return fig


def compute_top_issues(df: pd.DataFrame, top_n: int = 10):
    # Deprecated in favor of keywords.extract_keywords, kept if needed later
    tmp = df[df["sentiment_label"] == "Negative"].copy() if "sentiment_label" in df.columns else df.copy()
    return tmp.head(top_n)


def compute_competitor_comparison(df: pd.DataFrame):
    if "brand" not in df.columns:
        return px.bar(title="No brand column found")
    agg = df.groupby(["brand", "sentiment_label"]).size().reset_index(name="count")
    fig = px.bar(agg, x="brand", y="count", color="sentiment_label", barmode="group",
                 title="Competitor sentiment comparison")
    fig.update_layout(legend_title_text="Sentiment")
    return fig
