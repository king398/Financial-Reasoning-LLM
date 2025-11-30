from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from  tqdm import tqdm
# Load tokenizer & model via HF pipeline
model_name = "Mithilss/finance-llm"
tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model_name,

)
generator.push_to_hub("Mithilss/finance-llm-merged")

def create_prompts(example):
    return {"infer_prompt": [{"role": "user", "content": example['input']}]}


# Load dataset
validation_dataset = load_dataset("Mithilss/financial-training-v2")['validation']

# Add prompts
validation_dataset = validation_dataset.map(create_prompts)
labels = validation_dataset['Signal']
outputs = []
accurate = 0
for i,element in tqdm(enumerate(validation_dataset['infer_prompt'])):
    output = generator(element,max_new_tokens=128,
                    return_full_text=False, do_sample=False)
    if labels[i] == output[0]['generated_text']:
        accurate += 1
accuracy = accurate / len(validation_dataset)
print(f"Accuracy: {accuracy}")