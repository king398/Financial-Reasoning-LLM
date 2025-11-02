import json
from datasets import load_dataset
with open("dedup_groups.json", "r") as f:
    data = json.load(f)

data_set = set(data)
dataset = load_dataset("Mithilss/nasdaq-external-data-2018-onwards")
filter_dataset = dataset["train"].filter(lambda x: x["uuid"]  in data_set,num_proc=32)
filter_dataset.push_to_hub("Mithilss/nasdaq-external-data-2018-onwards-dedup")