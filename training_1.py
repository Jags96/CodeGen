import logging
import os
import time
from argparse import Namespace
from pathlib import Path
import csv
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
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorForSeq2Seq, AutoConfig, AutoModelForCausalLM

import os
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorForSeq2Seq



########## Parameters ##################

train_batch_size = 2
valid_batch_size = 2
shuffle_buffer = 10000
dataset_name_valid = "codeparrot/codeparrot-clean"
dataset_name_train = "codeparrot/codeparrot-clean"
seed = 42
seq_length = 1024

weight_decay = 0.1
lr_scheduler_type = "cosine"
num_warmup_steps = 750 
learning_rate = 2e-4
max_train_steps = 5000
max_eval_steps = 5000


gradient_accumulation_steps = 16
gradient_checkpointing = True
save_dir = './'
save_checkpoint_steps = 1024


model_name = 'PyModel_1'


## dataset and tokenization

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
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['content'])

#### constantlengthstrategry class ##########

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

################ DataLoader ######################################



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

train_dataloader, eval_dataloader  = create_dataloaders(dataset_name_valid,dataset_name_train,seq_length,shuffle_buffer,seed)


############## Initializing model ##################################

config_kwargs = {"vocab_size": len(tokenizer),
                 "scale_attn_by_layer_idx": True,
                 "reorder_and_upcast_attn": True}

# Load model with config and push to hub
config = AutoConfig.from_pretrained('gpt2', **config_kwargs)
config.n_layer = 6  # Reduced the number of layers
# config.n_embd = 384  # Reduced the hidden size (embedding dimension)
config.n_head = 6  # Reduced the number of attention heads
model = AutoModelForCausalLM.from_config(config)
model.save_pretrained(model_name)



############ Accelerator #####################

accelerator = Accelerator()
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

samples_per_step = accelerator.state.num_processes * train_batch_size
set_seed(seed)

############  Loading models #####################

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(save_dir)
if gradient_checkpointing:
    model.gradient_checkpointing_enable()



################## optimizers #####################


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


optimizer = AdamW(get_grouped_params(model), lr=learning_rate)
lr_scheduler = get_scheduler(name=lr_scheduler_type, optimizer=optimizer,
                             num_warmup_steps=num_warmup_steps,
                             num_training_steps=max_train_steps,)

accelerator.register_for_checkpointing(lr_scheduler)


def get_lr():
    return optimizer.param_groups[0]["lr"]


############## using accelerator for DistributedDataParallel

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)



############ training ##############################



model.train()
completed_steps = 0
max_eval_steps = -1



def evaluate(valid_batch_size,max_eval_steps):
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
    return loss.item(), perplexity.item()




# for step, batch in enumerate(train_dataloader, start=1):
#     print(f'current step:{step}')
#     loss = model(batch, labels=batch, use_cache=False).loss
#     loss = loss / gradient_accumulation_steps
#     accelerator.backward(loss)
#     if step % gradient_accumulation_steps == 0:
#         accelerator.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         completed_steps += 1
#     if step % save_checkpoint_steps == 0:
#         eval_loss, perplexity = evaluate(valid_batch_size,max_eval_steps)
#         accelerator.wait_for_everyone()
#         unwrapped_model = accelerator.unwrap_model(model)
#         unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
#         if accelerator.is_main_process:
#             hf_repo.push_to_hub(commit_message=f"step {step}")
#         model.train()
#     if completed_steps >= max_train_steps:
#         break






# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()



save_dir = "./"  # Save to the current directory
metrics_file = "training_metrics.csv"  # Metrics file to store the training logs
training_metrics = []

# Save header to CSV if the file is empty
if not os.path.exists(metrics_file):
    with open(metrics_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'eval_loss', 'perplexity'])
# Training loop
for step, batch in enumerate(train_dataloader, start=1):
    logger.info(f"Current step: {step}")
    
    # Forward pass
    outputs = model(batch, labels=batch, use_cache=False)
    loss = outputs.loss
    loss = loss / gradient_accumulation_steps  # Gradient accumulation
    accelerator.backward(loss)  # Backpropagate
    
    if step % gradient_accumulation_steps == 0:
        accelerator.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()  # Update optimizer
        lr_scheduler.step()  # Update learning rate scheduler
        optimizer.zero_grad()  # Zero gradients
        completed_steps += 1  # Increment completed steps
    
    # Save checkpoint
    if step % save_checkpoint_steps == 0:
        eval_loss, perplexity = evaluate(valid_batch_size, max_eval_steps)
        

        logger.info(f"Step {step}: Eval Loss = {eval_loss}, Perplexity = {perplexity}")
        

        logger.info(f"Step {step}: Train Loss = {loss.item()}, Eval Loss = {eval_loss}, Perplexity = {perplexity}")
        

        training_metrics.append({
            'step': step,
            'train_loss': loss.item(),
            'eval_loss': eval_loss,
            'perplexity': perplexity
        })
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir)
        
        
        with open(metrics_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'train_loss', 'eval_loss', 'perplexity'])
            writer.writerow({'step': step, 'train_loss': loss.item(), 'eval_loss': eval_loss, 'perplexity': perplexity}) # Save metrics to CSV after every checkpoint
        
        logger.info(f"Model saved to {save_dir} at step {step}")
        
        model.train() 
    
    if completed_steps >= max_train_steps:
        logger.info("Training complete. Maximum steps reached.")
        break