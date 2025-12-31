#!/usr/bin/env python3
"""
FunctionGemma Fine-Tuning Script for OpenShift Commands

This script fine-tunes Google's FunctionGemma (270M parameters) to convert
natural language into OpenShift/Kubernetes CLI commands.

Tested on: Red Hat OpenShift AI with Tesla T4 GPU

Usage:
    python finetune_functiongemma.py

Requirements:
    pip install transformers>=4.51.0 peft datasets accelerate

Author: Nirjhar Jajodia
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ============================================================
# GPU Driver Fix (Required for OpenShift AI with CUDA 13)
# ============================================================
def fix_gpu_driver():
    """Fix CUDA driver mismatch on OpenShift AI clusters."""
    try:
        import ctypes
        ctypes.CDLL('/lib64/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)
        print("✅ GPU driver fix applied")
    except Exception as e:
        print(f"⚠️ GPU driver fix not needed or failed: {e}")

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "google/functiongemma-270m-it"
OUTPUT_DIR = "./finetuned-functiongemma"

# Training hyperparameters (optimized for stability)
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
MAX_LENGTH = 64

# LoRA configuration (conservative for stability)
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# ============================================================
# Training Data
# ============================================================
TRAINING_DATA = [
    # GET PODS - many variations
    "User: show all pods\nCommand: oc get pods\n",
    "User: list pods\nCommand: oc get pods\n",
    "User: show pods\nCommand: oc get pods\n",
    "User: get pods\nCommand: oc get pods\n",
    "User: display pods\nCommand: oc get pods\n",
    "User: what pods are running\nCommand: oc get pods\n",
    "User: pods\nCommand: oc get pods\n",
    
    # GET DEPLOYMENTS
    "User: list deployments\nCommand: oc get deployments\n",
    "User: show deployments\nCommand: oc get deployments\n",
    "User: get deployments\nCommand: oc get deployments\n",
    "User: deployments\nCommand: oc get deployments\n",
    "User: what deployments exist\nCommand: oc get deployments\n",
    
    # SCALE
    "User: scale nginx to 5\nCommand: oc scale deployment nginx --replicas=5\n",
    "User: scale nginx to 3\nCommand: oc scale deployment nginx --replicas=3\n",
    "User: scale nginx to 10\nCommand: oc scale deployment nginx --replicas=10\n",
    "User: scale web to 5\nCommand: oc scale deployment web --replicas=5\n",
    "User: scale api to 2\nCommand: oc scale deployment api --replicas=2\n",
    "User: set nginx replicas 5\nCommand: oc scale deployment nginx --replicas=5\n",
    
    # DELETE POD
    "User: delete pod test\nCommand: oc delete pod test\n",
    "User: delete pod nginx\nCommand: oc delete pod nginx\n",
    "User: delete pod web\nCommand: oc delete pod web\n",
    "User: remove pod test\nCommand: oc delete pod test\n",
    "User: kill pod broken\nCommand: oc delete pod broken\n",
    
    # LOGS
    "User: get logs nginx\nCommand: oc logs nginx\n",
    "User: logs nginx\nCommand: oc logs nginx\n",
    "User: show logs nginx\nCommand: oc logs nginx\n",
    "User: logs api\nCommand: oc logs api\n",
    "User: get logs web\nCommand: oc logs web\n",
    
    # DESCRIBE
    "User: describe pod nginx\nCommand: oc describe pod nginx\n",
    "User: describe pod web\nCommand: oc describe pod web\n",
    "User: pod details nginx\nCommand: oc describe pod nginx\n",
    
    # NAMESPACES
    "User: get namespaces\nCommand: oc get namespaces\n",
    "User: list namespaces\nCommand: oc get namespaces\n",
    "User: show namespaces\nCommand: oc get namespaces\n",
    
    # NODES
    "User: get nodes\nCommand: oc get nodes\n",
    "User: list nodes\nCommand: oc get nodes\n",
    "User: show nodes\nCommand: oc get nodes\n",
    
    # CREATE NAMESPACE
    "User: create namespace dev\nCommand: oc create namespace dev\n",
    "User: create namespace test\nCommand: oc create namespace test\n",
    "User: create namespace prod\nCommand: oc create namespace prod\n",
    
    # SERVICES
    "User: get services\nCommand: oc get services\n",
    "User: list services\nCommand: oc get services\n",
    "User: show services\nCommand: oc get services\n",
    
    # ROUTES
    "User: get routes\nCommand: oc get routes\n",
    "User: list routes\nCommand: oc get routes\n",
    "User: show routes\nCommand: oc get routes\n",
    
    # EVENTS
    "User: get events\nCommand: oc get events\n",
    "User: show events\nCommand: oc get events\n",
    
    # SECRETS
    "User: get secrets\nCommand: oc get secrets\n",
    "User: list secrets\nCommand: oc get secrets\n",
    
    # RESTART
    "User: restart deployment nginx\nCommand: oc rollout restart deployment nginx\n",
    "User: restart deployment api\nCommand: oc rollout restart deployment api\n",
    "User: restart nginx\nCommand: oc rollout restart deployment nginx\n",
]


def tokenize_function(text: str, tokenizer) -> dict:
    """Tokenize a single example with proper label masking."""
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    # Only compute loss on non-padding tokens
    tokens["labels"] = [
        t if t != tokenizer.pad_token_id else -100 
        for t in tokens["input_ids"]
    ]
    return tokens


def main():
    print("=" * 60)
    print("FunctionGemma Fine-Tuning for OpenShift Commands")
    print("=" * 60)
    
    # --------------------------------------------------------
    # Step 0: Fix GPU Driver (OpenShift AI specific)
    # --------------------------------------------------------
    fix_gpu_driver()
    
    # --------------------------------------------------------
    # Step 1: Check GPU Availability
    # --------------------------------------------------------
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device_map = "cuda"
        dtype = torch.float32  # float32 for stability
    else:
        print("⚠️ No GPU found, using CPU")
        device_map = "auto"
        dtype = torch.float32
    
    # --------------------------------------------------------
    # Step 2: Load Model and Tokenizer
    # --------------------------------------------------------
    print("\n📥 Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map
    )
    
    print(f"✅ Model loaded!")
    print(f"   Parameters: {model.num_parameters():,}")
    
    # --------------------------------------------------------
    # Step 3: Configure LoRA
    # --------------------------------------------------------
    print("\n⚙️ Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],  # Conservative: fewer modules
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
    # Step 4: Prepare Training Data
    # --------------------------------------------------------
    print("\n📊 Preparing training data...")
    
    dataset = Dataset.from_list([{"text": t} for t in TRAINING_DATA])
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x["text"], tokenizer),
        remove_columns=["text"]
    )
    
    # Verify labels
    sample_labels = [l for l in tokenized_dataset[0]["labels"] if l != -100]
    print(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
    print(f"   Sample has {len(sample_labels)} trainable tokens")
    
    # --------------------------------------------------------
    # Step 5: Configure Training
    # --------------------------------------------------------
    print("\n🎯 Configuring training...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=25,
        save_strategy="no",
        fp16=False,  # Disabled for stability
        report_to="none",
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
    # Step 6: Train
    # --------------------------------------------------------
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    trainer.train()
    
    # Get final loss
    losses = [l['loss'] for l in trainer.state.log_history if 'loss' in l]
    print("-" * 60)
    print(f"✅ Training complete!")
    print(f"   Loss: {losses[0]:.2f} → {losses[-1]:.2f}")
    
    # --------------------------------------------------------
    # Step 7: Save Model
    # --------------------------------------------------------
    print("\n💾 Saving model...")
    
    peft_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Model saved to {OUTPUT_DIR}")
    
    # --------------------------------------------------------
    # Step 8: Test Model
    # --------------------------------------------------------
    print("\n🧪 Testing fine-tuned model...")
    print("-" * 60)
    
    test_prompts = [
        "list pods",
        "get services",
        "list deployments",
        "get nodes",
        "show namespaces",
    ]
    
    peft_model.eval()
    
    for prompt in test_prompts:
        input_text = f"User: {prompt}\nCommand:"
        inputs = tokenizer(input_text, return_tensors="pt").to(peft_model.device)
        
        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        command = response.split("Command:")[-1].strip().split("\n")[0] if "Command:" in response else "?"
        
        print(f"📝 Input:  '{prompt}'")
        print(f"🤖 Output: {command}")
        print()
    
    print("-" * 60)
    print("🎉 All done!")
    print(f"   Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
