# =========================================
# GPT-2 Fine-Tuning on a Custom Dataset
# =========================================

# Install required libraries if not already installed:
# pip install transformers datasets torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# ----------------------------
# 1. LOAD PRE-TRAINED GPT-2
# ----------------------------
# We start with GPT-2, a transformer-based language model from OpenAI.
# Hugging Face's Transformers library makes it easy to load and fine-tune.
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set PAD token to EOS
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# ----------------------------
# 2. PREPARE CUSTOM DATASET
# ----------------------------
# This example uses a text file ("custom_dataset.txt") containing your training data.
# You can also use Hugging Face's 'datasets' to load from CSV, JSON, etc.
dataset = load_dataset("text", data_files={"train": "custom_dataset.txt"})

# ----------------------------
# 3. TOKENIZATION
# ----------------------------
# Tokenization converts text into numerical IDs GPT-2 understands.
# We also set truncation and padding to make all sequences the same length.
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=20
    )
    # For causal LM, labels are the same as input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# GPT-2 doesn't have a pad token by default, so we set it to EOS (end-of-sequence).
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# ----------------------------
# 4. TRAINING ARGUMENTS
# ----------------------------
# We define hyperparameters and settings for fine-tuning.
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",     # Where to save the model
    overwrite_output_dir=True,
    num_train_epochs=3,                # Number of passes over the dataset
    per_device_train_batch_size=2,     # Adjust based on your GPU/CPU
    save_steps=500,                    # Save model every 500 steps
    save_total_limit=2,                # Only keep last 2 checkpoints
    logging_dir="./logs",              # Directory for logs
    logging_steps=100,
    prediction_loss_only=True,
)

# ----------------------------
# 5. TRAINER
# ----------------------------
# The Trainer class handles the training loop for us.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

import os
os.environ["WANDB_DISABLED"] = "true"

# ----------------------------
# 6. TRAINING
# ----------------------------
print("Starting training...")
trainer.train()



# ----------------------------
# 7. SAVE THE MODEL
# ----------------------------
# This allows you to load it later for text generation.
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

print("Model fine-tuning complete!")

## wandb: af64d75e80bf6bba6eefdcd52cf2258e73bd3f9d
