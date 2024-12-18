import os

DEVICES = ",".join(map(str, range(17, 32)))
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICES

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from trl import SFTTrainer
from datasets import load_dataset

# 1. Load the base model and tokenizer
model_name = "/home/students/stalakol/llama-models/downloads"  # Make sure you have access to this model
# model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device_map = {
    "": [3, 4, 5]  # Distribute across CUDA devices 3, 4, and 5
}

# 2. Configure 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
)

# 3. Prepare model for training with PEFT
model = prepare_model_for_kbit_training(model)

# 4. Configure LoRA parameters
lora_config = LoraConfig(
    r=16,  # Rank of the low-rank adaptation
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. Prepare the model with LoRA
model = get_peft_model(model, lora_config)

# 6. Load and preprocess the dataset
# def format_instruction(sample):
#     """Format dataset to instruction-response format"""
#     return {
#         "text": f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
#     }

def format_instruction(sample):
    """Format dataset to instruction-response format"""
    instruction = sample['instruction']
    code_input = sample['input']  # Assuming the input is available as 'input' in the sample
    
    prompt = f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the Task.

### Task:
{instruction}

### Input:
{code_input}

### Response:
"""
    
    return {"text": prompt}

# Load Python code instruction dataset
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")  # Replace with actual dataset
dataset = dataset.map(format_instruction)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,
    report_to="tensorboard"
)

# 8. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    formatting_func=lambda x: x['text'],
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args
)

# 9. Train the model
trainer.train()

# 10. Save the fine-tuned model
trainer.save_model("./python-code-llama2-7b")

# 11. Optional: Push to Hugging Face Hub
# trainer.push_to_hub()

# 12. Inference example
def generate_python_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
print(generate_python_code("Write a function to calculate fibonacci sequence"))
