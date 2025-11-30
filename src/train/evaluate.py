from transformers import pipeline
from datasets import load_dataset
from vllm import LLM, SamplingParams

llm = LLM(model="Mithilss/finance-llm", tensor_parallel_size=2)
params = SamplingParams(temperature=0.0)
tokenizer = llm.get_tokenizer()


def create_prompts(input):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": input['input']}], add_generation_prompt=True,
                                           tokenize=False)
    return {"infer_prompt": prompt}


validation_dataset = load_dataset("Mithilss/financial-training-v2")['validation']

validation_dataset = validation_dataset.map(create_prompts)
outputs = llm.generate(validation_dataset['infer_prompt'][:25], params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
