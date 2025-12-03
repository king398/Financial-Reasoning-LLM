"""
Utilities for transforming LLM trade predictions into a normalized portfolio
equity curve that the benchmarking toolkit can consume.

This script mirrors the ad-hoc steps we ran manually:
    1. Load the Hugging Face validation split to get ground-truth price paths.
    2. Align every row with the LLM's BUY/SELL/HOLD prediction.
    3. Convert each prediction into a 3-day return (longing BUY, shorting SELL).
    4. Equal-weight all signals that share the same `End_Date` and compound the
       resulting daily returns into an equity curve that starts at 1.0.

Example:
    python build_portfolio.py \
        --predictions finance_llm_predictions.csv \
        --output portfolio_equity.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import pandas as pd
from datasets import load_dataset


HF_DATASET_NAME = "Mithilss/financial-training-v2"
HF_SPLIT = "validation"


def _load_predictions(predictions_csv: str) -> pd.DataFrame:
    """Read the LLM predictions CSV and sanitize the `prediction` column."""

    preds = pd.read_csv(predictions_csv)
    if "prompt" not in preds.columns or "prediction" not in preds.columns:
        raise ValueError("Predictions CSV must contain 'prompt' and 'prediction'.")

    preds = preds.rename(columns={"prompt": "input_text"})
    preds["prediction"] = preds["prediction"].astype(str).str.strip().str.upper()
    return preds[["input_text", "prediction"]]


def _load_validation_split() -> pd.DataFrame:
    """Download the Hugging Face validation split and convert it to pandas."""

    split = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)
    df = split.to_pandas()
    # The `input` column is the bare prompt without chat template tokens.
    df = df.rename(columns={"input": "input_text"})
    df["End_Date"] = pd.to_datetime(df["End_Date"])
    return df


def _align_predictions(
    hf_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """Join the predictions to the HF rows; return the merged frame and miss count."""

    merged = hf_df.merge(predictions_df, on="input_text", how="left")
    missing_predictions = merged["prediction"].isna().sum()
    if missing_predictions:
        # Leave explicit breadcrumbs when we have to impute signals.
        merged["prediction"] = merged["prediction"].fillna("HOLD")
    return merged, int(missing_predictions)


def _signal_to_return(prediction: str, change: float) -> float:
    """Map each BUY/SELL/HOLD prediction to the corresponding return."""

    if prediction == "BUY":
        return change
    if prediction == "SELL":
        return -change
    # HOLD or unrecognized instructions keep capital in cash for that interval.
    return 0.0


def build_portfolio_equity(predictions_csv: str) -> pd.DataFrame:
    """Construct the normalized daily equity curve implied by the LLM signals."""

    predictions = _load_predictions(predictions_csv)
    hf_df = _load_validation_split()
    merged, missing = _align_predictions(hf_df, predictions)
    if missing:
        print(
            f"[build_portfolio] Warning: {missing} HF rows lacked predictions; defaulted to HOLD.",
            file=sys.stderr,
        )

    price_change = (merged["Future_Close"] - merged["End_Close"]) / merged["End_Close"]
    merged["signal_return"] = [
        _signal_to_return(pred, change)
        for pred, change in zip(merged["prediction"], price_change, strict=True)
    ]

    # Equal-weight all signals sharing the same End_Date so every stock in play
    # contributes equally to the portfolio on that day.
    by_date = merged.groupby("End_Date")["signal_return"].mean()

    # Forward-fill over a business-day calendar to keep the benchmarking utilities happy.
    calendar = pd.bdate_range(start=by_date.index.min(), end=by_date.index.max())
    daily_returns = by_date.reindex(calendar, fill_value=0.0)

    wealth = (1.0 + daily_returns).cumprod()
    return pd.DataFrame({"date": wealth.index, "equity": wealth.values})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LLM BUY/SELL/HOLD predictions into a normalized equity curve.",
    )
    parser.add_argument(
        "--predictions",
        default="finance_llm_predictions.csv",
        help="Path to the CSV containing prompt,label,prediction columns.",
    )
    parser.add_argument(
        "--output",
        default="portfolio_equity.csv",
        help="Destination CSV for the generated equity curve.",
    )
    args = parser.parse_args()

    equity = build_portfolio_equity(args.predictions)
    equity.to_csv(args.output, index=False)
    print(f"[build_portfolio] Wrote {len(equity)} rows to {args.output}")


if __name__ == "__main__":
    main()
