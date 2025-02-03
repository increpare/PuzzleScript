import argparse
import glob
import os
import json
import shutil
os.environ["BITSANDBYTES_NOWELCOME"] = "true"

from datasets import Dataset
import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, PrinterCallback, TrainerCallback, set_seed
from trl import SFTTrainer


def gpu_utilization():
    return torch.cuda.memory_allocated() / 1024 / 1024


class CustomTQDMCallback(TrainerCallback):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=state.max_steps, desc=f"Training {self.model_name}")
        self.progress_bar.update(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)

        if len(state.log_history) > 0:
            most_recent_logs = state.log_history[-1].copy()
            most_recent_logs.pop("step", None)
            most_recent_logs["GPU"] = f"{gpu_utilization()}MB"

            self.progress_bar.set_postfix(most_recent_logs)


parser = argparse.ArgumentParser()

# Model arguments
parser.add_argument('--model', type=str, help="The name of the model to use")
parser.add_argument('--bits', type=int, default=None, choices=[4, 8], help="The number of bits to use for quantization")

# LoRA arguments
parser.add_argument('--lora', action='store_true', help="Whether to use low-rank adaptation for finetuning")
parser.add_argument('--lora_alpha', type=int, default=16, help="The alpha parameter for LoRA")
parser.add_argument('--lora_dropout', type=float, default=0.05, help="The dropout rate for LoRA")
parser.add_argument('--lora_r', type=int, default=64, help="The r parameter for LoRA")

# Training arguments
parser.add_argument('--num_epochs', type=int, default=3, help="Number of epochs to train for")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--block_size', type=int, default=1024)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Initial learning rate (after the potential warmup period) to use")
parser.add_argument('--optimizer', type=str, default='adamw_torch', choices=['adamw_torch', 'paged_adamw_32bit'])
parser.add_argument('--warmup_ratio', type=float, default=0.03, help="Proportion of total steps to warm up the learning rate for")
parser.add_argument('--max_grad_norm', type=float, default=0.3, help="Maximum gradient norm to clip to")
parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay to use for the optimizer")

# Dataset arguments
parser.add_argument("--dataset_type", type=str, default="expanded", choices=["base", "realised", "expanded", "instruct", "fitm"])
parser.add_argument("--data_categories", type=str, nargs='+', default=["board"])
parser.add_argument("--data_subcategories", type=str, nargs='+', default="")
parser.add_argument("--mask_names", action='store_true', help="Whether to mask the names of the games and pieces in the dataset")
parser.add_argument("--train_prop", type=float, default=0.8)
parser.add_argument("--val_prop", type=float, default=0.2)
parser.add_argument("--use_val_set", action='store_true', help="Whether to use a specific set of games for validation")

# Run arguments
parser.add_argument('--overwrite', action='store_true', help="Whether to overwrite the save directory if it already exists")
parser.add_argument('--save_dir', type=str, default='./logs/llm_test', help="Directory to save the model")
parser.add_argument('--save_freq', type=int, default=500, help="How often to save the model during training")
parser.add_argument('--save_limit', type=int, default=3, help="Number of model checkpoints to keep during training")
parser.add_argument('--seed', type=int, default=42, help="Random seed to use")

args = parser.parse_args()

# Check to see if the save directory already exists
if os.path.exists(args.save_dir):
    if not args.overwrite:
        response = input(f"Save directory {args.save_dir} already exists. Overwrite? (y/n) ")
        if response.lower() != "y":
            exit("Exiting without overwriting save directory...")

    print("Overwriting save directory...")
    shutil.rmtree(args.save_dir)

# Save run arguments
os.makedirs(args.save_dir, exist_ok=True)
json.dump(args.__dict__, open(os.path.join(args.save_dir, "run_args.json"), "w"))

# Set the random seed
set_seed(args.seed)

# Apply quantization
if args.bits is not None:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=args.bits == 8,
        load_in_4bit=args.bits == 4
    )

    device_map = "auto"
    torch_dtype = torch.bfloat16

else:
    device_map = None
    quantization_config = None
    torch_dtype = None

# Load the model
# model_name = "microsoft/Phi-3.5-mini-instruct"  # or similar
model_name = "gpt2"  # or similar
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "right"

# Set the training arguments
training_args=TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.num_epochs,
    max_steps=-1,
    
    optim=args.optimizer,
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,
    # weight_decay=args.weight_decay,
    # max_grad_norm=args.max_grad_norm,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    
    logging_steps=1,
    logging_dir=args.save_dir,
    
    output_dir=args.save_dir,
    save_strategy="steps",
    save_steps=args.save_freq,
    save_total_limit=args.save_limit,
    disable_tqdm=True,
)

# Define the LoraConfig
if args.lora:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# Load the dataset
# val_set = VALIDATION_GAMES if args.use_val_set else None

# 1. Load data
text_files = glob.glob("scraped_puzzles/*.txt")
all_texts = []
for path in text_files:
    with open(path, "r", encoding="utf-8") as f:
        all_texts.append(f.read())
dataset = Dataset.from_dict({"text": all_texts})

print(f"Dataset size: {len(dataset)}")

# Instantiate the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    callbacks=[CustomTQDMCallback(model_name=args.model)],
    max_seq_length=args.block_size,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    peft_config=peft_config,
)

# Remove the default PrinterCallback (since we're rolling its functionality into our CustomTQDMCallback)
trainer.remove_callback(PrinterCallback)

# Train and save
trainer.train()
trainer.save_model(os.path.join(args.save_dir, "final_model"))