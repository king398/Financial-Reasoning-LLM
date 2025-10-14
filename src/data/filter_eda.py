from datasets import load_dataset
from collections import Counter
from datetime import datetime
ds = load_dataset("Mithilss/finance-data-2020")['train']
def is_after(example,year=2020):
    try:
        date_obj = datetime.strptime(example["Date"], "%Y-%m-%d %H:%M:%S %Z")
        return {"year":int(date_obj.year)}
    except Exception:
        return {"year":0}  # In case of malformed date strings
ds = ds.map(is_after, num_proc=16)
value_counts = Counter(ds['year'])
for value, count in value_counts.most_common(10): # Show top 10 most common values
    print(f"'{value}': {count}")

