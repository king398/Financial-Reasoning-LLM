import pandas as pd
from datasets import load_dataset, Value
from intervaltree import IntervalTree
from datetime import datetime

ds = load_dataset("Mithilss/nasdaq-external-data-2018-onwards-dedup")['train']
stock_list = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK.B", "JPM", "V",
    "LLY", "JNJ", "WMT", "MA", "XOM", "BAC", "PG", "ORCL", "HD", "DIS",
    "CMCSA", "ADBE", "NFLX", "INTC", "PFE", "ABT", "CRM", "T", "PEP", "COST",
    "CVX", "ABBV", "NKE", "ACN", "MCD", "TXN", "QCOM", "DHR", "NEE", "MDT",
    "HON", "LIN", "AMGN", "BMY", "IBM", "MS", "UPS", "SCHW", "PM", "COP"
]
ds = ds.filter(lambda x: x['Stock_symbol'] in stock_list)

start_date = "2019-01-01"
end_date = "2023-12-31"
date_ranges = pd.date_range(start=start_date, end=end_date, freq="7D")

tree = IntervalTree.from_tuples([
    (date_ranges[i], date_ranges[i + 1])
    for i in range(len(date_ranges) - 1)
])


def mapping(example):
    example["Date"] = datetime.strptime(example["Date"], "%Y-%m-%d %H:%M:%S %Z")
    matches = list(tree[example["Date"]])
    if matches:
        interval = matches[0]
        example["Interval"] = f"{interval.begin.date()}_{interval.end.date()}"
    else:
        example["Interval"] = None
    return example


ds = ds.map(mapping)
print(ds[0]['Interval'])
print(ds[0]['Date'])
