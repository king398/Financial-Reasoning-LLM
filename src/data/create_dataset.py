import os
import json
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from huggingface_hub import login


START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
TOP_SYMBOLS_PATH = "src/data/top_500_symbols.json"  # path to symbol list
OUTPUT_HF_DATASET = "kushalt31/unified-nasdaq-ohlcv-2018-2025"  # your dataset name


if not os.path.exists(TOP_SYMBOLS_PATH):
    raise FileNotFoundError(f"Cannot find {TOP_SYMBOLS_PATH}. Please provide a valid path.")

with open(TOP_SYMBOLS_PATH, "r") as f:
    symbols_data = json.load(f)


if isinstance(symbols_data, dict):
    symbols = list(symbols_data.keys())
elif isinstance(symbols_data, list):
    symbols = symbols_data
else:
    raise ValueError("Invalid format in top_500_symbols.json — must be list or dict.")

symbols = [s.strip().upper() for s in symbols if s and s.lower() != "nan"]

print(f"✅ Loaded {len(symbols)} stock symbols.")


all_data = []

for symbol in tqdm(symbols, desc="Fetching stock data"):
    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)

        if df.empty:
            print(f"⚠️ Skipping {symbol}: empty data.")
            continue

        # If columns are multi-index (e.g. ('Price', 'Close')), flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df.reset_index(inplace=True)
        df["Symbol"] = symbol
        df = df[["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]]
        df.rename(columns={"Date": "Datetime"}, inplace=True)

        all_data.append(df)

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        continue



if not all_data:
    raise RuntimeError("No data collected for any symbols.")

combined_df = pd.concat(all_data, ignore_index=True)
combined_df.dropna(inplace=True)
combined_df["Datetime"] = pd.to_datetime(combined_df["Datetime"])

print(f"✅ Combined dataset shape: {combined_df.shape}")
print(combined_df.head())


hf_dataset = Dataset.from_pandas(combined_df)


hf_dataset.save_to_disk("unified_ohlcv_dataset")


try:
    hf_dataset.push_to_hub(OUTPUT_HF_DATASET)
    print(f"🚀 Successfully uploaded dataset to {OUTPUT_HF_DATASET}")
except Exception as e:
    print(f"⚠️ Upload failed: {e}")
