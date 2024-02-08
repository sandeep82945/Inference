#installation pip install sacremoses
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import pipeline, set_seed
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

set_seed(42) #Line sets the random seed to ensure that the generated text is reproducible.


print(response)

def main():
    response = generator("COVID-19 is", max_length=20, num_return_sequences=1, do_sample=True)

if __name__ == '__main__':
    