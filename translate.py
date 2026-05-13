import torch
from transformers import MarianMTModel, MarianTokenizer

device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

translator_model = "Helsinki-NLP/opus-mt-en-ru"
tokenizer = MarianTokenizer.from_pretrained(translator_model)
translator = MarianMTModel.from_pretrained(translator_model).to(device)

example = "i love cats and dogs"
tokens = tokenizer(example, return_tensors="pt").to(device)
translated = translator.generate(**tokens)
result = tokenizer.decode(translated[0], skip_special_tokens=True)
print(result)
