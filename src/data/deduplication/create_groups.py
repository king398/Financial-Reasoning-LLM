import numpy as np
from tqdm import tqdm
from autofaiss import build_index
import faiss
import os
import json

# Load data
data = np.load("embeddings.npz")
document_ids = data["document_ids"]
embeddings = data["embedding"].astype("float32")

# Load or build index
if os.path.exists("knn.index"):
    print("Loading existing index...")
    index = faiss.read_index("knn.index")
else:
    print("Building new FAISS index...")
    index, index_infos = build_index(
        embeddings,
        save_on_disk=True,
        max_index_memory_usage="64GB"
    )
    faiss.write_index(index, "knn.index")

# Search for top 5 similar vectors for each embedding
k = 5
batch_size = 10000
results = {}

for start in tqdm(range(0, len(embeddings), batch_size)):
    end = min(start + batch_size, len(embeddings))
    batch = embeddings[start:end]
    distances, indices = index.search(batch, k)

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        query_id = str(document_ids[start + i])
        top_results = [
            {"id": str(document_ids[j]), "distance": float(dist)}
            for j, dist in zip(idxs, dists)
        ]
        results[query_id] = top_results

# Save to JSON
with open("top5_similarities.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Saved top-5 similarity results to top5_similarities.json")
