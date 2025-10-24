from datasets import load_dataset
from embed import embed_documents,load_model
import numpy as np
ds = load_dataset("Mithilss/nasdaq-external-data-2018-onwards")['train']
documents = ds['Article']
document_ids = ds['uuid']
tokenizer, model  = load_model()
embedding = embed_documents(documents=documents,model=model,tokenizer=tokenizer,batch_size=64).numpy()
np.savez("embeddings.npz", embedding=embedding, document_ids=document_ids)