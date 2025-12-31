#!/usr/bin/env python3
"""
FunctionGemma Fine-Tuning Script for OpenShift Commands

This script fine-tunes Google's FunctionGemma (270M parameters) to convert
natural language into OpenShift/Kubernetes CLI commands.

Usage:
    python finetune_functiongemma.py

Requirements:
    pip install transformers peft datasets accelerate

Author: Nirjhar Jajodia
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "google/functiongemma-270m-it"
OUTPUT_DIR = "./finetuned-functiongemma"
TRAINING_DATA_FILE = "training_data.json"

# Training hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
MAX_LENGTH = 64

# LoRA configuration
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def load_training_data(filepath: str) -> list:
    """Load training data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def tokenize_function(example: dict, tokenizer) -> dict:
    """Tokenize a single example."""
    text = f"User: {example['input']}\nCommand: oc {example['output']}"
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    print("=" * 60)
    print("FunctionGemma Fine-Tuning for OpenShift Commands")
    print("=" * 60)
    
    # --------------------------------------------------------
    # Step 1: Load Model and Tokenizer
    # --------------------------------------------------------
    print("\n📥 Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    print(f"✅ Model loaded!")
    print(f"   Parameters: {model.num_parameters():,}")
    
    # --------------------------------------------------------
    # Step 2: Configure LoRA
    # --------------------------------------------------------
    print("\n⚙️ Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    
    print(f"✅ LoRA configured!")
    print(f"   Total params:     {total:,}")
    print(f"   Trainable params: {trainable:,}")
    print(f"   Trainable %:      {100 * trainable / total:.2f}%")
    
    # --------------------------------------------------------
    # Step 3: Load and Prepare Training Data
    # --------------------------------------------------------
    print("\n📊 Loading training data...")
    
    training_data = load_training_data(TRAINING_DATA_FILE)
    print(f"   Loaded {len(training_data)} examples")
    
    # Tokenize dataset
    dataset = Dataset.from_list(training_data)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=dataset.column_names
    )
    
    print(f"✅ Dataset prepared!")
    
    # --------------------------------------------------------
    # Step 4: Configure Training
    # --------------------------------------------------------
    print("\n🎯 Configuring training...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,  # CPU doesn't support fp16
        report_to="none",
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print(f"✅ Trainer configured!")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    # --------------------------------------------------------
    # Step 5: Train
    # --------------------------------------------------------
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    trainer.train()
    
    print("-" * 60)
    print("✅ Training complete!")
    
    # --------------------------------------------------------
    # Step 6: Save Model
    # --------------------------------------------------------
    print("\n💾 Saving model...")
    
    peft_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Model saved to {OUTPUT_DIR}")
    
    # --------------------------------------------------------
    # Step 7: Test Model
    # --------------------------------------------------------
    print("\n🧪 Testing fine-tuned model...")
    print("-" * 60)
    
    test_prompts = [
        "show me all pods",
        "scale frontend to 3 replicas",
        "create new project staging",
        "show all deployments",
    ]
    
    peft_model.eval()
    
    for prompt in test_prompts:
        input_text = f"User: {prompt}\nCommand: oc"
        inputs = tokenizer(input_text, return_tensors="pt").to(peft_model.device)
        
        outputs = peft_model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        command = response.split("Command: oc")[-1].strip() if "Command: oc" in response else response
        
        print(f"📝 Input:  '{prompt}'")
        print(f"🤖 Output: oc {command}")
        print()
    
    print("-" * 60)
    print("🎉 All done!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print("   You can now use this model to convert natural language to oc commands!")


if __name__ == "__main__":
    main()

