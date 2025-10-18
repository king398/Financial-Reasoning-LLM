from datasets import load_dataset
from datetime import datetime

# Load dataset
ds = load_dataset("Mithilss/nasdaq-external-data-processed", num_proc=16)['train']

# --- 1️⃣ Count total rows ---
print("Total rows:", len(ds))

def is_after(example):
    try:
        dt = datetime.strptime(example["Date"], "%Y-%m-%d %H:%M:%S %Z")
        return dt.year > 2018
    except Exception as e:
        return False

filtered = ds.filter(is_after, num_proc=16)
print("Rows after 2019:", len(filtered))
filtered.push_to_hub("Mithilss/nasdaq-external-data-2018-onwards")