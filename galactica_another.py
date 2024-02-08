# pip install accelerate
import torch
from transformers import AutoTokenizer, OPTForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b")
#model = OPTForCausalLM.from_pretrained("facebook/galactica-30b", device_map="auto", torch_dtype=torch.float16)
model = OPTForCausalLM.from_pretrained("facebook/galactica-30b", device_map="auto")

prompt = """Imagine you are a research scientist, analyse the following paper:  
```
Introduction: Socioeconomic factors have been recognized by the WHO as determinants of health, and it is important to consider these factors in decision making to curb existing inequality in vaccination for SARS-CoV-2, which causes COVID-19. Objective: We aimed to determine whether there is a correlation between socioeconomic factors and vaccination worldwide and measure inequality. Method: A study of secondary sources was carried out to assess inequality in vaccination against COVID-19 worldwide and its association with socioeconomic factors. For this assessment, 169 countries were chosen from January 2020 to March 2022 using LibreOffice and JASP 0.16.1.10. Several mathematical models and statistical tests were used, including a normality test, an analysis of frequencies and proportions, a Kruskal–Wallis test, Spearman’s correlations, a Lorenz curve, a Concentration Index, and a slope.
```
Possible future research ideas/work for the paper are Answer:"""

def inference(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids, max_new_tokens= 1500)
    return tokenizer.decode(outputs[0])


def main():
    input_text = "The Transformer architecture [START_REF]"
    print(inference(prompt))
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    # outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0]))

if __name__ == '__main__':
    main()