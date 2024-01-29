import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline


base_model_id = "mistralai/Mixtral-8x7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

prompt = "As a data scientist, can you explain the concept of regularization in machine learning?"

sequences = pipe(
    prompt,
    do_sample=True,
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=1,
)
print(sequences[0]['generated_text'])