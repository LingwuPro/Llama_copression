from transformers import LlamaTokenizer

model_name = "yahma/llama-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name)
text = "Hello World"
inputs_dict = tokenizer(text)
print(inputs_dict)
