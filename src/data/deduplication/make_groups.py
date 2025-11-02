import json
from datasets import load_dataset
with open("dedup_groups.json", "r") as f:
    data = json.load(f)
print(data[0])
keep_id = [x[0] for x in data]
dataset = load_dataset("Mithilss/nasdaq-external-data-2018-onwards")
print(keep_id[0])
print(dataset["train"][0]['uuid'])
filter_dataset = dataset["train"].filter(lambda x: x["uuid"]  in keep_id,num_proc=32)
filter_dataset.push_to_hub("Mithilss/nasdaq-external-data-2018-onwards-dedup")