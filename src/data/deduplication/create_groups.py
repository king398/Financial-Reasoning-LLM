import numpy as np
import torch
from tqdm import tqdm
from autofaiss import build_index

data = np.load("embeddings.npz")

document_ids =  data["document_ids"]
embeddings = data["embedding"]
index, index_infos = build_index(embeddings, save_on_disk=False)

for i in tqdm(range(len(embeddings))):
    _, I = index.search(embeddings[i], 5)

