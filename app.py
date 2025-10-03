import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment==3.3.2"])


import streamlit as st
import pandas as pd
from pathlib import Path

from src.data_loader import load_dataset
from src.sentiment import SentimentAnalyzer
from src.keywords import extract_keywords
from src.analytics import (
    compute_sentiment_over_time,
    compute_top_issues,
    compute_competitor_comparison,
)

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

DATA_DIR = Path("data")
DEFAULT_FILE = DATA_DIR / "sample_reviews.csv"

@st.cache_data(show_spinner=False)
def get_data(file_path: Path):
    return load_dataset(file_path)

@st.cache_resource(show_spinner=False)
def get_analyzer():
    return SentimentAnalyzer()


def sidebar_controls(df: pd.DataFrame):
    st.sidebar.header("Controls")
    brands = sorted([b for b in df["brand"].dropna().unique()]) if "brand" in df.columns else []
    selected_brands = st.sidebar.multiselect("Brands", options=brands, default=brands[:2] if brands else [])
    date_col = "date" if "date" in df.columns else None
    if date_col and df[date_col].notna().any():
        min_date = pd.to_datetime(df[date_col]).min()
        max_date = pd.to_datetime(df[date_col]).max()
        if pd.isna(min_date) or pd.isna(max_date):
            date_range = None
        else:
            date_range = st.sidebar.date_input("Date range", value=[min_date.date(), max_date.date()])
    else:
        date_range = None
    return selected_brands, date_range


def main():
    st.title("AI-Powered Market Research & Sentiment Analysis")

    uploaded = st.file_uploader("Upload CSV with columns: date, text, brand", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if DEFAULT_FILE.exists():
            df = get_data(DEFAULT_FILE)
        else:
            st.info("Upload a CSV to get started, or add `data/sample_reviews.csv`.")
            st.stop()

    # Normalize columns
    expected_cols = {"text"}
    if not expected_cols.issubset(set(df.columns)):
        st.error("CSV must include at least a 'text' column. Optional: 'date', 'brand'.")
        st.stop()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    analyzer = get_analyzer()

    with st.spinner("Scoring sentiment..."):
        df = analyzer.add_sentiment(df)

    selected_brands, date_range = sidebar_controls(df)

    # Filter
    if selected_brands and "brand" in df.columns:
        df = df[df["brand"].isin(selected_brands)]
    if date_range and "date" in df.columns and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["date"] >= start) & (df["date"] <= end)]

    if df.empty:
        st.warning("No data after applying filters.")
        st.stop()

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Records", len(df))
    with c2:
        pos_rate = (df["sentiment_label"] == "Positive").mean() if len(df) else 0
        st.metric("Positive %", f"{pos_rate*100:.1f}%")
    with c3:
        neg_rate = (df["sentiment_label"] == "Negative").mean() if len(df) else 0
        st.metric("Negative %", f"{neg_rate*100:.1f}%")

    # Charts
    st.subheader("Sentiment trend over time")
    trend_fig = compute_sentiment_over_time(df)
    st.plotly_chart(trend_fig, use_container_width=True)

    st.subheader("Top issues customers complain about")
    with st.spinner("Extracting keywords..."):
        issues_df = extract_keywords(df, label_filter="Negative")
    st.dataframe(issues_df, use_container_width=True, height=320)

    st.subheader("Competitor comparison")
    comp_fig = compute_competitor_comparison(df)
    st.plotly_chart(comp_fig, use_container_width=True)

    with st.expander("Managerial Insights", expanded=True):
        st.markdown("- Marketing: Increase focus on themes with high negative keywords (e.g., delivery).")
        st.markdown("- Product: Prioritize fixes tied to frequent complaint keywords (e.g., quality).")
        st.markdown("- Competitive: Highlight strengths vs competitors where positive share is higher.")
        st.markdown("- CX: If negative spikes align with dates, audit operations and communications that week.")

    st.caption("Powered by VADER + YAKE. Extendable to transformers.")


if __name__ == "__main__":
    main()
