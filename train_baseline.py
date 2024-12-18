import os
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorForSeq2Seq

# Load the dataset
dataset = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# Simplified remove_columns function
def remove_columns(examples):
    return {k: v for k, v in examples.items() if k == 'content'}

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['content'], padding="max_length", truncation=True)

# Apply remove_columns and tokenize
dataset = dataset.map(remove_columns, batched=True)

import logging
import os
import time
from argparse import Namespace
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.utils import ProjectConfiguration
# from arguments import TrainingArguments
from datasets import load_dataset
from huggingface_hub import Repository
from torch.optim import AdamW
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, get_scheduler, set_seed
import csv
import os
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorForSeq2Seq



accelerator = Accelerator()
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

class ConstantLengthDataset(IterableDataset):  ### can directly feed iterable
    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        tokenized=False,
    ):
        self.tokenizer = tokenizer   
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.epoch = 0
        self.infinite = infinite
        self.current_size = 0
        self.tokenized = tokenized

        if self.tokenized:
            self.max_buffer_size = seq_length * num_of_sequences
            self.content_field = "input_ids"
        else:
            self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
            self.content_field = "content"

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            if self.tokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield torch.tensor(input_ids)

    def shuffle(self, buffer_size=1000):
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)


cld = ConstantLengthDataset(
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        tokenized=False,
    )

train_batch_size = 2
valid_batch_size = 2
shuffle_buffer = 10000
dataset_name_valid = "codeparrot/codeparrot-clean-train"
dataset_name_train = "codeparrot/codeparrot-clean-train"
seed = 42
seq_length = 1024

def create_dataloaders(dataset_name_valid,dataset_name_train,seq_length,shuffle_buffer,seed):
    train_data = load_dataset(dataset_name_train, split="train", streaming = True)
    train_data = train_data.shuffle(buffer_size=shuffle_buffer, seed=seed)
    valid_data = load_dataset(dataset_name_valid, split="train", streaming = True)
    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=True, seq_length=seq_length, tokenized=False
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=seq_length, tokenized=False
    )
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size)
    return train_dataloader, eval_dataloader

train_dataloader, eval_dataloader = create_dataloaders(dataset_name_valid,dataset_name_train,seq_length,shuffle_buffer,seed)


from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorForSeq2Seq, AutoConfig, AutoModelForCausalLM
config = AutoConfig.from_pretrained("gpt2")


config.n_layer = 6  # Reduce the number of layers
config.n_embd = 384  # Reduce the hidden size (embedding dimension)
config.n_head = 6  # Reduce the number of attention heads

# Initialize the model with the modified config
model = AutoModelForCausalLM.from_config(config)


# Save model to the hub
model.save_pretrained('experiment')

accelerator = Accelerator()
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

samples_per_step = accelerator.state.num_processes * train_batch_size
set_seed(seed)


weight_decay = 0.1

def get_grouped_params(model, no_decay=["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]



lr_scheduler_type = "cosine"
num_warmup_steps = 250 
learning_rate = 3e-4
max_train_steps = 5000
max_eval_steps = 500

optimizer = AdamW(get_grouped_params(model), lr=learning_rate)
lr_scheduler = get_scheduler(name=lr_scheduler_type, optimizer=optimizer,
                             num_warmup_steps=num_warmup_steps,
                             num_training_steps=max_train_steps,)


accelerator.register_for_checkpointing(lr_scheduler)


def get_lr():
    return optimizer.param_groups[0]["lr"]

### using accelerator for Distributed Data Parallel

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)


max_eval_steps = 5000
def evaluate(valid_batch_size,max_eval_steps):
    print(f'called model.eval()')
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(valid_batch_size)
        losses.append(accelerator.gather(loss))
        if max_eval_steps > 0 and step >= max_eval_steps:
            break
    losses = torch.cat(losses)
    loss = losses[: eval_dataloader.dataset.current_size].mean()
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    print(f'returned loss, perplexity')
    return loss.item(), perplexity.item()

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()



save_dir = "./"  
metrics_file = "training_metrics_notebook.csv"  


gradient_accumulation_steps = 16 
max_train_steps = 50000
save_checkpoint_steps = 1024  
completed_steps = 0

training_metrics = []
if not os.path.exists(metrics_file):
    with open(metrics_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'eval_loss', 'perplexity'])

        
for step, batch in enumerate(tqdm(train_dataloader, desc="Training", unit="batch", dynamic_ncols=True), start=1):
    outputs = model(batch, labels=batch, use_cache=False)
    loss = outputs.loss
    loss = loss / gradient_accumulation_steps  # Gradient accumulation
    accelerator.backward(loss)  # Backpropagate
    
    if step % gradient_accumulation_steps == 0:
        logger.info(f"Current step: {step}")
        accelerator.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()  # Update optimizer
        lr_scheduler.step()  # Update learning rate scheduler
        optimizer.zero_grad()  # Zero gradients
        completed_steps += 1  # Increment completed steps
    
    # Save checkpoint
    if step % save_checkpoint_steps == 0:
        # Evaluate and get loss, perplexity
        eval_loss, perplexity = evaluate(valid_batch_size, max_eval_steps)
        

        logger.info(f"Step {step}: Eval Loss = {eval_loss}, Perplexity = {perplexity}")
        

        logger.info(f"Step {step}: Train Loss = {loss.item()}, Eval Loss = {eval_loss}, Perplexity = {perplexity}")
        

        training_metrics.append({
            'step': step,
            'train_loss': loss.item(),
            'eval_loss': eval_loss,
            'perplexity': perplexity
        })
        
        # Save the model to the current directory
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir)
        
        # Save metrics to CSV after every checkpoint
        with open(metrics_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'train_loss', 'eval_loss', 'perplexity'])
            writer.writerow({'step': step, 'train_loss': loss.item(), 'eval_loss': eval_loss, 'perplexity': perplexity})
        
        logger.info(f"Model saved to {save_dir} at step {step}")
        
        model.train()  # Set model back to training mode
    
    # Check if max training steps reached
    if completed_steps >= max_train_steps:
        logger.info("Training complete. Maximum steps reached.")
        break




