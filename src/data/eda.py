from datasets import load_dataset
from scrape import get_website
from datetime import datetime
ds = load_dataset("Mithilss/finance-data").shuffle(seed=42)
def is_after(example,year=2020):
    try:
        date_obj = datetime.strptime(example["Date"], "%Y-%m-%d %H:%M:%S %Z")
        return int(date_obj.year) >= year
    except Exception:
        return False  # In case of malformed date strings

filtered_ds = ds["train"].filter(is_after, num_proc=16)
print(filtered_ds)
filtered_ds.push_to_hub("Mithilss/finance-data-2020")