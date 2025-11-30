from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import pandas as pd
import torch
import numpy as np

# -------------------------
# Load tokenizer & model
# -------------------------
model_name = "Mithilss/finance-llm-merged"
tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    return_full_text=False,
    dtype=torch.float16,
)

# -------------------------
# Create prompt
# -------------------------
def create_prompts(example):
    return {"infer_prompt": [{"role": "user", "content": example['input']}]}

# -------------------------
# Load dataset
# -------------------------
validation_dataset = load_dataset("Mithilss/financial-training-v2")['validation'].select(range(10))
validation_dataset = validation_dataset.map(create_prompts)

labels = validation_dataset['Signal']
prompts = validation_dataset['infer_prompt']

# -------------------------
# Batched inference
# -------------------------
batch_size = 4
predictions = []

for start in tqdm(range(0, len(prompts), batch_size)):
    batch = prompts[start:start + batch_size]

    outputs = generator(
        batch,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    for out in outputs:
        predictions.append(out[0]['generated_text'])

# -------------------------
# Compute metrics
# -------------------------
df = pd.DataFrame({
    "prompt": [p[0]["content"] for p in prompts],
    "label": labels,
    "prediction": predictions
})

accuracy = (df["label"] == df["prediction"]).mean()
macro_f1 = f1_score(df["label"], df["prediction"], average="macro")
precision = precision_score(df["label"], df["prediction"], average="macro", zero_division=0)
recall = recall_score(df["label"], df["prediction"], average="macro", zero_division=0)

cm = confusion_matrix(df["label"], df["prediction"])
class_report = classification_report(df["label"], df["prediction"])

# -------------------------
# Print metrics
# -------------------------
print("\n========== METRICS ==========")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Macro F1:      {macro_f1:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", class_report)

# -------------------------
# Save predictions CSV
# -------------------------
df.to_csv("finance_llm_predictions.csv", index=False)

# -------------------------
# Save metrics report
# -------------------------
with open("finance_llm_metrics.txt", "w") as f:
    f.write("========== METRICS ==========\n")
    f.write(f"Accuracy:      {accuracy:.4f}\n")
    f.write(f"Macro F1:      {macro_f1:.4f}\n")
    f.write(f"Precision:     {precision:.4f}\n")
    f.write(f"Recall:        {recall:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm) + "\n\n")

    f.write("Classification Report:\n")
    f.write(class_report + "\n")

print("Saved metrics → finance_llm_metrics.txt")
print("Saved predictions → finance_llm_predictions.csv")
