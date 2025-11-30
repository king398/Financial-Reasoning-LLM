from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
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
    torch_dtype=torch.float16,
    device_map="auto",   # or device=0 if single GPU
)

# -------------------------
# Create prompt
# -------------------------
def create_prompts(example):
    return {"infer_prompt": [{"role": "user", "content": example["input"]}]}

# -------------------------
# Load dataset
# -------------------------
validation_dataset = load_dataset("Mithilss/financial-training-v2")["validation"]
validation_dataset = validation_dataset.map(create_prompts)

labels = list(validation_dataset["Signal"])
prompts = list(validation_dataset["infer_prompt"])

# quick sanity check
assert len(labels) == len(prompts), (
    f"len(labels)={len(labels)} != len(prompts)={len(prompts)}"
)

batch_size = 4

# Hugging Face pipeline can take the whole list with internal batching
# We'll manually tqdm over indices just to get a progress bar, but
# only one underlying pipeline call will be used per batch.
predictions = []

for start in tqdm(range(0, len(prompts), batch_size), total=(len(prompts) + batch_size - 1) // batch_size):
    batch = prompts[start:start + batch_size]

    outputs = generator(
        batch,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # outputs is a list of lists: [[{generated_text: ...}], ...]
    for out in outputs:
        # Be defensive: handle weird cases
        if not out or "generated_text" not in out[0]:
            predictions.append(None)
        else:
            predictions.append(out[0]["generated_text"])

# Final sanity check
assert len(predictions) == len(labels), (
    f"len(predictions)={len(predictions)} != len(labels)={len(labels)}"
)

# -------------------------
# Normalize labels
# -------------------------
def normalize_signal(x):
    if x is None:
        return "UNKNOWN"
    s = str(x).strip().upper()
    # adapt these to your exact label strings
    if s.startswith("BUY"):
        return "BUY"
    if s.startswith("SELL"):
        return "SELL"
    if s.startswith("HOLD"):
        return "HOLD"
    return "UNKNOWN"

norm_labels = [normalize_signal(l) for l in labels]
norm_preds = [normalize_signal(p) for p in predictions]

df = pd.DataFrame({
    "prompt": [p[0]["content"] for p in prompts],
    "label": norm_labels,
    "prediction": norm_preds,
})

# Make absolutely sure sklearn only sees strings
df["label"] = df["label"].astype(str)
df["prediction"] = df["prediction"].astype(str)

# Optionally, you can restrict metrics to the 3 main classes:
classes = ["BUY", "SELL", "HOLD"]

# -------------------------
# Compute metrics
# -------------------------
accuracy = (df["label"] == df["prediction"]).mean()

macro_f1 = f1_score(
    df["label"],
    df["prediction"],
    average="macro",
    labels=classes
)

precision = precision_score(
    df["label"],
    df["prediction"],
    average="macro",
    labels=classes,
    zero_division=0,
)

recall = recall_score(
    df["label"],
    df["prediction"],
    average="macro",
    labels=classes,
    zero_division=0,
)

cm = confusion_matrix(df["label"], df["prediction"], labels=classes)
class_report = classification_report(df["label"], df["prediction"], labels=classes)

# -------------------------
# Print metrics
# -------------------------
print("\n========== METRICS ==========")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Macro F1:      {macro_f1:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print("\nConfusion Matrix (BUY/SELL/HOLD order):\n", cm)
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

    f.write("Confusion Matrix (BUY/SELL/HOLD order):\n")
    f.write(np.array2string(cm) + "\n\n")

    f.write("Classification Report:\n")
    f.write(class_report + "\n")

print("Saved metrics → finance_llm_metrics.txt")
print("Saved predictions → finance_llm_predictions.csv")
