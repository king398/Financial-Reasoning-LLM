import json

# Load JSON
with open("dedup_groups.json", "r") as f:
    data = json.load(f)

# Sum up all counts 
total = sum(data.values())

print(f"Total samples across top 500 symbols: {total}")
