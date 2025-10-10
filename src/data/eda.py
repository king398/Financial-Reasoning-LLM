from datasets import load_dataset
from scrape import get_website
ds = load_dataset("Mithilss/finance-data").shuffle(seed=42)
print(ds['train']['Stock_symbol'].value_counts())