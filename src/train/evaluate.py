from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

# Load tokenizer & model via HF pipeline
model_name = "Mithilss/finance-llm"
tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    max_new_tokens=128,
    return_full_text=False,
)

def create_prompts(example):

    return {"infer_prompt": [{"role": "user", "content": example['input']}]}

# Load dataset
validation_dataset = load_dataset("Mithilss/financial-training-v2")['validation']

# Add prompts
validation_dataset = validation_dataset.map(create_prompts)

outputs = generator(validation_dataset['infer_prompt'][:25])
for i in outputs:
    output = i["generated_text"]
    print(f"Generated text: {output!r}")
