import os
import glob
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# 1. Load data
text_files = glob.glob("scraped_puzzles/*.txt")
all_texts = []
for path in text_files:
    with open(path, "r", encoding="utf-8") as f:
        all_texts.append(f.read())
dataset = Dataset.from_dict({"text": all_texts})


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Tokenize
model_name = "microsoft/Phi-3.5-mini-instruct"  # or similar
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    tokens = tokenizer(examples["text"], truncation=True, max_length=100)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# 3. Configure training
model = AutoModelForCausalLM.from_pretrained(model_name)
training_args = TrainingArguments(
    output_dir="./finetuned_phi",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    eval_strategy="no",
    gradient_accumulation_steps=1,
    fp16=True,
)

# 4. Fine-tune
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

og_compute_loss = trainer.compute_loss

# Add logging to inspect the inputs
def compute_loss_with_logging(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    inputs.pop('num_items_in_batch', None)
    logging.info(f"Input: {inputs.keys()}")
    loss = og_compute_loss(model, inputs, return_outputs, num_items_in_batch)
    return loss

trainer.compute_loss = compute_loss_with_logging.__get__(trainer, Trainer)

breakpoint()

import torch
torch.cuda.empty_cache()

trainer.train()
