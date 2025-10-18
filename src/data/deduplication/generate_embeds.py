from datasets import load_dataset
from embed import embed_documents,load_model
ds = load_dataset("Mithilss/nasdaq-external-data-processed")['train']
documents = ds['Article']
tokenizer, model  = load_model()
embed_documents(documents=documents,model=model,tokenizer=tokenizer,batch_size=64)
