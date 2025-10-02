# AI-Powered Market Research & Sentiment Analysis

A Streamlit-based dashboard for analyzing customer sentiment and keywords from reviews/tweets to compare brands and track issues over time.

## Features
- Sentiment analysis (Positive/Neutral/Negative) using VADER or optional transformers
- Keyword extraction using YAKE
- Trends over time and competitor comparisons
- Top issues and topics

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Data
- Place CSVs in `data/` with columns like: `date`, `text`, `brand` (brand optional; you can pick a single brand if not present).
- Sample included: `data/sample_reviews.csv`.

## Notes
- For small/medium projects, VADER is fast and works well on reviews/tweets.
- If you want transformer models, you can extend `sentiment.py` with Hugging Face pipelines.
