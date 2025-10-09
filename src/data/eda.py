from datasets import load_dataset
from scrape import get_website
ds = load_dataset("Mithilss/finance-data",streaming=True)
url = next(iter(ds['train']))['Url']
get_website(url)