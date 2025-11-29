from transformers import pipeline
from datasets import load_dataset

validation_dataset = load_dataset("Mithilss/financial-training")['validation']

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="Mithilss/unsloth_training_checkpoints", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
