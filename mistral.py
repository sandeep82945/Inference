#https://github.com/run-llama/llama_index/blob/main/docs/examples/llm/mistralai.ipynb

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("t5-11b")
model = AutoModelWithLMHead.from_pretrained("t5-11b")


prompt = "Can you explain the concept of regularization in machine learning?"

sequences = pipeline(
    prompt,
    do_sample=True,
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=1,
)
print(sequences[0]['generated_text'])