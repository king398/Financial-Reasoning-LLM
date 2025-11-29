import pandas as pd
from datasets import load_dataset, Dataset
from intervaltree import IntervalTree
from datetime import datetime
from pandas.tseries.offsets import BDay
from tqdm import tqdm
from transformers import AutoTokenizer

# 1️⃣ Load datasets
ds = load_dataset("Mithilss/nasdaq-external-data-2018-onwards")['train']
price_dataset = load_dataset("kushalt31/unified-nasdaq-ohlcv-2018-2025")['train']
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
# 2️⃣ Stock list
stock_list = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK.B", "JPM", "V",
    "LLY", "JNJ", "WMT", "MA", "XOM", "BAC", "PG", "ORCL", "HD", "DIS",
    "CMCSA", "ADBE", "NFLX", "INTC", "PFE", "ABT", "CRM", "T", "PEP", "COST",
    "CVX", "ABBV", "NKE", "ACN", "MCD", "TXN", "QCOM", "DHR", "NEE", "MDT",
    "HON", "LIN", "AMGN", "BMY", "IBM", "MS", "UPS", "SCHW", "PM", "COP"
]

# Filter articles
ds = ds.filter(lambda x: x['Stock_symbol'] in stock_list, num_proc=32)
ds = ds.filter(lambda x: x['Textrank_summary'] != 'nan')

# 3️⃣ Build 7-day intervals
start_date = "2018-01-01"
end_date = "2023-12-31"
date_ranges = pd.date_range(start=start_date, end=end_date, freq="3D")

tree = IntervalTree.from_tuples([
    (date_ranges[i], date_ranges[i + 1])
    for i in range(len(date_ranges) - 1)
])


# 4️⃣ Mapping functions
def mapping(example):
    example["Date"] = datetime.strptime(example["Date"], "%Y-%m-%d %H:%M:%S %Z")
    matches = list(tree[example["Date"]])
    if matches:
        interval = matches[0]
        example["Interval"] = f"{interval.begin.date()}_{interval.end.date()}"
        example["end_date"] = interval.end.date()
    else:
        example["Interval"] = None
        example["end_date"] = None
    return example


def create_prompt(example):
    example['text'] = f"""Article Title: {example['Article_title']}
Article Text: {example['Textrank_summary']}"""
    return example


# Apply functions
ds = ds.map(mapping, num_proc=32)
ds = ds.map(create_prompt, num_proc=32)
ds = ds.filter(lambda x: x['Interval'] is not None)

# 5️⃣ Convert to pandas
df = ds.to_pandas()

# 6️⃣ Prepare price data
px_ds = price_dataset.filter(lambda x: x['Symbol'] in stock_list, num_proc=32)
px = px_ds.to_pandas()

px['date'] = pd.to_datetime(px['Datetime']).dt.normalize()
px = (
    px.sort_values(['Symbol', 'date'])
    .drop_duplicates(['Symbol', 'date'])
)

# 7️⃣ Build (Stock_symbol, Interval, end_date, future_date)
labels = (
    df[['Stock_symbol', 'Interval', 'end_date']]
    .dropna()
    .drop_duplicates()
)
labels['end_date'] = pd.to_datetime(labels['end_date'])
labels['future_date'] = (labels['end_date'] + BDay(3)).dt.normalize()  # 3 business days ahead

# 8️⃣ Merge-asof to get the next available price row (3 days later)
future_merge = pd.merge_asof(
    labels.sort_values('future_date'),
    px.rename(columns={'Symbol': 'Stock_symbol'}).sort_values('date'),
    left_on='future_date',
    right_on='date',
    by='Stock_symbol',
    direction='forward'
).rename(columns={'Close': 'Future_Close', 'date': 'Future_Date'})

# 9️⃣ Merge-asof again to get the close price at end_date
end_merge = pd.merge_asof(
    labels.sort_values('end_date'),
    px.rename(columns={'Symbol': 'Stock_symbol'}).sort_values('date'),
    left_on='end_date',
    right_on='date',
    by='Stock_symbol',
    direction='backward'
).rename(columns={'Close': 'End_Close', 'date': 'End_Date_Actual'})

# 1️⃣0️⃣ Combine both (end + future close)
merged = pd.merge(
    end_merge[['Stock_symbol', 'Interval', 'end_date', 'End_Close']],
    future_merge[['Stock_symbol', 'Interval', 'Future_Date', 'Future_Close']],
    on=['Stock_symbol', 'Interval'],
    how='left'
)


# 1️⃣1️⃣ Add signal
def signal_func(row, threshold=0.001):  # ±0.5% tolerance
    if pd.isna(row['Future_Close']) or pd.isna(row['End_Close']):
        return None
    change = (row['Future_Close'] - row['End_Close']) / row['End_Close']
    if change > threshold:
        return "BUY"
    elif change < -threshold:
        return "SELL"
    else:
        return "HOLD"


merged.dropna()
merged['Signal'] = merged.apply(signal_func, axis=1)

# 1️⃣2️⃣ Build LLM prompt dataset
prompt = f"""
You are an expert financial analyst and market reasoning model.

Task:
Given the following news titles and summaries about a single stock from the past 7 days,
decide whether an investor should **buy**, **sell**, or **hold** the stock in the next 3 days.

Use overall sentiment, event type (earnings, innovation, controversy, etc.), and market psychology to decide.
News and titles:
"""

dataset = []
for stock in tqdm(stock_list):
    stock_df = df[df["Stock_symbol"] == stock]
    for interval, group_df in stock_df.groupby("Interval"):
        texts = "\n".join(group_df["text"].tolist())
        end_date = group_df['end_date'].iloc[0]
        row = merged[(merged['Stock_symbol'] == stock) & (merged['Interval'] == interval)]
        if row.empty:
            continue
        future_close = row['Future_Close'].iloc[0]
        end_close = row['End_Close'].iloc[0]
        signal = row['Signal'].iloc[0]
        closing_price = f"Closing Price on the final date - {future_close:.2f}"
        messages = [
            {"role": "user", "content": prompt + texts + closing_price},
            {"role": "assistant", "content": signal}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        dataset.append({
            "Stock": stock,
            "Interval": interval,
            "End_Date": end_date,
            "End_Close": end_close,
            "Future_Close": future_close,
            "Signal": signal,
            "prompt": text,
            "input":prompt + texts + closing_price,
        })

final_df = pd.DataFrame(dataset)
final_ds = Dataset.from_pandas(final_df)
final_ds.push_to_hub("Mithilss/financial-training")
# Optional: save for later
final_df.to_csv("stock_news_signal_dataset.csv", index=False)
