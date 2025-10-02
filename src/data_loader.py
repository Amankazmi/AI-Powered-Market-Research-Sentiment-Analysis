from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(file_path: Path | str) -> pd.DataFrame:
    """Load a CSV dataset and normalize common columns.

    Expected columns:
      - text: required
      - date: optional; will be parsed to datetime
      - brand: optional; free text brand label
    """
    path = Path(file_path)
    df = pd.read_csv(path)
    # Standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Keep only likely useful columns if present
    keep = [c for c in ["date", "text", "brand"] if c in df.columns]
    if keep:
        df = df[keep + [c for c in df.columns if c not in keep]]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df
