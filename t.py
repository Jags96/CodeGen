import torch 
# import config.json
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorForSeq2Seq, AutoConfig, AutoModelForCausalLM
# config = AutoConfig.from_pretrained("gpt2")


# config.n_layer = 6  # Reduce the number of layers
# config.n_embd = 384  # Reduce the hidden size (embedding dimension)
# config.n_head = 6  # Reduce the number of attention heads

# Initialize the model with the modified config
config = AutoConfig.from_pretrained("./config.json")

model = AutoModelForCausalLM.from_pretrained(
    "./model.safetensors", config = config)


tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_text = "Write a function of sum of list of numbers"

inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)

decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)