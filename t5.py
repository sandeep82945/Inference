import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("t5-11b")
model = T5ForConditionalGeneration.from_pretrained("t5-11b")



prompt = "hello"

pipeline = pipeline(
    "text-generation",
    do_sample=True,
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=1,
)

sequences = pipeline(prompt)
print(sequences[0]['generated_text'])