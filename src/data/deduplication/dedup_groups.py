import json
import networkx as nx
from tqdm import tqdm
# Load data
with open("top5_similarities.json", "r") as f:
    data = json.load(f)

# Build a similarity graph
G = nx.Graph()
threshold = 0.98  # adjust for your deduplication sensitivity

for source_id, neighbors in tqdm(data.items()):
    G.add_node(source_id)
    for entry in neighbors:
        target_id = entry["id"]
        distance = entry["distance"]
        if distance >= threshold:
            G.add_edge(source_id, target_id)

# Extract connected components as groups
groups = list(nx.connected_components(G))

# Convert to list of lists (for saving or inspection)
groups_list = [list(group) for group in groups]

# Example output
for i, g in enumerate(groups_list[:5]):
    print(f"Group {i+1}: {g}")

# Optional: save for later use
import json
with open("dedup_groups.json", "w") as f:
    json.dump(groups_list, f, indent=2)
