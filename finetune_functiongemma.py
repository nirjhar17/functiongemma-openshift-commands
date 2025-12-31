#!/usr/bin/env python3
"""
FunctionGemma Fine-Tuning Script for OpenShift Commands
========================================================

This script fine-tunes Google's FunctionGemma (270M parameters) to convert
natural language into OpenShift/Kubernetes CLI commands.

WHAT IS FUNCTIONGEMMA?
----------------------
FunctionGemma is a tiny AI model (270M parameters) from Google designed for
"function calling" - converting natural language into structured commands.
Think of it as teaching a small AI to understand what you want and call the
right function with the right arguments.

Example:
  User says: "Turn off the lights"
  Model outputs: smart_home(action="lights_off")

We're training it to do:
  User says: "Show me all pods"
  Model outputs: oc get pods

WHY FINE-TUNING?
----------------
The base model knows HOW to call functions, but doesn't know OpenShift commands.
Fine-tuning teaches it our specific commands while keeping its general abilities.

WHAT IS LoRA?
-------------
LoRA (Low-Rank Adaptation) is a clever trick:
- Instead of changing ALL 270 million parameters (expensive!)
- We add tiny "adapter" layers and only train those
- Result: Train 0.14% of parameters, get 95% of the benefit
- Like putting sticky notes on a textbook instead of rewriting it

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
# STEP 0: GPU DRIVER FIX
# ============================================================
# 
# WHAT'S THE PROBLEM?
# On OpenShift AI, the container has CUDA 12.8 libraries, but the
# GPU node might have a newer driver (CUDA 13.0). This causes a
# conflict and PyTorch can't see the GPU.
#
# THE FIX:
# We manually load the correct NVIDIA driver library BEFORE importing
# torch. This tells Python to use the node's driver instead of the
# container's old compatibility libraries.
#
# Think of it like: "Hey Python, use THIS driver, not that old one!"
#
def fix_gpu_driver():
    """Fix CUDA driver mismatch on OpenShift AI clusters."""
    try:
        import ctypes
        # Load the correct driver from the node (not the container)
        ctypes.CDLL('/lib64/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)
        print("✅ GPU driver fix applied")
    except Exception as e:
        print(f"⚠️ GPU driver fix not needed or failed: {e}")


# ============================================================
# CONFIGURATION
# ============================================================
#
# These are the "knobs" you can turn to adjust training.
# I've set them to values that work well after lots of testing!
#

MODEL_NAME = "google/functiongemma-270m-it"  # The base model from HuggingFace
OUTPUT_DIR = "./finetuned-functiongemma"     # Where to save the trained model

# TRAINING HYPERPARAMETERS
# -----------------------
# NUM_EPOCHS: How many times to go through all training data
#   - Too few (1-5): Model doesn't learn enough
#   - Too many (100+): Model "memorizes" instead of "learning"
#   - Sweet spot: 20-50 for our dataset size
NUM_EPOCHS = 30

# BATCH_SIZE: How many examples to process at once
#   - Bigger = faster, but needs more GPU memory
#   - Tesla T4 (15GB) can handle 4-8 easily for this small model
BATCH_SIZE = 4

# LEARNING_RATE: How big of a step to take when learning
#   - Too high (1e-3): Model weights "explode", outputs garbage
#   - Too low (1e-6): Takes forever to learn anything
#   - Sweet spot for LoRA: 1e-5 to 1e-4
LEARNING_RATE = 5e-5

# MAX_LENGTH: Maximum tokens (words/subwords) per example
#   - Our commands are short, 64 is plenty
MAX_LENGTH = 64

# LoRA CONFIGURATION
# -----------------
# LORA_RANK (r): Size of the adapter matrices
#   - Higher = more capacity to learn, but slower
#   - 4-16 is typical, 8 is a good default
LORA_RANK = 8

# LORA_ALPHA: Scaling factor for LoRA weights
#   - Usually set to 2x the rank
#   - Higher = stronger effect of fine-tuning
LORA_ALPHA = 16

# LORA_DROPOUT: Randomly "turn off" some connections during training
#   - Prevents overfitting (model memorizing instead of learning)
#   - 0.05-0.2 is typical
LORA_DROPOUT = 0.1


# ============================================================
# TRAINING DATA
# ============================================================
#
# This is what we're teaching the model!
# Format: "User: <natural language>\nCommand: <oc command>\n"
#
# TIPS FOR GOOD TRAINING DATA:
# 1. Include MULTIPLE variations of the same command
#    - "show pods", "list pods", "get pods" all → "oc get pods"
# 2. Be consistent with format
# 3. More examples = better learning (50+ recommended)
# 4. Cover the commands you actually want to use
#
TRAINING_DATA = [
    # ----- GET PODS -----
    # Notice how we have 7 different ways to ask for pods!
    # This helps the model understand that all these mean the same thing.
    "User: show all pods\nCommand: oc get pods\n",
    "User: list pods\nCommand: oc get pods\n",
    "User: show pods\nCommand: oc get pods\n",
    "User: get pods\nCommand: oc get pods\n",
    "User: display pods\nCommand: oc get pods\n",
    "User: what pods are running\nCommand: oc get pods\n",
    "User: pods\nCommand: oc get pods\n",
    
    # ----- GET DEPLOYMENTS -----
    "User: list deployments\nCommand: oc get deployments\n",
    "User: show deployments\nCommand: oc get deployments\n",
    "User: get deployments\nCommand: oc get deployments\n",
    "User: deployments\nCommand: oc get deployments\n",
    "User: what deployments exist\nCommand: oc get deployments\n",
    
    # ----- SCALE (more complex - has arguments) -----
    # These are harder for the model because it needs to:
    # 1. Understand "scale X to Y"
    # 2. Put X in the right place
    # 3. Put Y in --replicas=Y
    "User: scale nginx to 5\nCommand: oc scale deployment nginx --replicas=5\n",
    "User: scale nginx to 3\nCommand: oc scale deployment nginx --replicas=3\n",
    "User: scale nginx to 10\nCommand: oc scale deployment nginx --replicas=10\n",
    "User: scale web to 5\nCommand: oc scale deployment web --replicas=5\n",
    "User: scale api to 2\nCommand: oc scale deployment api --replicas=2\n",
    "User: set nginx replicas 5\nCommand: oc scale deployment nginx --replicas=5\n",
    
    # ----- DELETE POD -----
    "User: delete pod test\nCommand: oc delete pod test\n",
    "User: delete pod nginx\nCommand: oc delete pod nginx\n",
    "User: delete pod web\nCommand: oc delete pod web\n",
    "User: remove pod test\nCommand: oc delete pod test\n",
    "User: kill pod broken\nCommand: oc delete pod broken\n",
    
    # ----- LOGS -----
    "User: get logs nginx\nCommand: oc logs nginx\n",
    "User: logs nginx\nCommand: oc logs nginx\n",
    "User: show logs nginx\nCommand: oc logs nginx\n",
    "User: logs api\nCommand: oc logs api\n",
    "User: get logs web\nCommand: oc logs web\n",
    
    # ----- DESCRIBE -----
    "User: describe pod nginx\nCommand: oc describe pod nginx\n",
    "User: describe pod web\nCommand: oc describe pod web\n",
    "User: pod details nginx\nCommand: oc describe pod nginx\n",
    
    # ----- NAMESPACES -----
    "User: get namespaces\nCommand: oc get namespaces\n",
    "User: list namespaces\nCommand: oc get namespaces\n",
    "User: show namespaces\nCommand: oc get namespaces\n",
    
    # ----- NODES -----
    "User: get nodes\nCommand: oc get nodes\n",
    "User: list nodes\nCommand: oc get nodes\n",
    "User: show nodes\nCommand: oc get nodes\n",
    
    # ----- CREATE NAMESPACE -----
    "User: create namespace dev\nCommand: oc create namespace dev\n",
    "User: create namespace test\nCommand: oc create namespace test\n",
    "User: create namespace prod\nCommand: oc create namespace prod\n",
    
    # ----- SERVICES -----
    "User: get services\nCommand: oc get services\n",
    "User: list services\nCommand: oc get services\n",
    "User: show services\nCommand: oc get services\n",
    
    # ----- ROUTES -----
    "User: get routes\nCommand: oc get routes\n",
    "User: list routes\nCommand: oc get routes\n",
    "User: show routes\nCommand: oc get routes\n",
    
    # ----- EVENTS -----
    "User: get events\nCommand: oc get events\n",
    "User: show events\nCommand: oc get events\n",
    
    # ----- SECRETS -----
    "User: get secrets\nCommand: oc get secrets\n",
    "User: list secrets\nCommand: oc get secrets\n",
    
    # ----- RESTART (rollout) -----
    "User: restart deployment nginx\nCommand: oc rollout restart deployment nginx\n",
    "User: restart deployment api\nCommand: oc rollout restart deployment api\n",
    "User: restart nginx\nCommand: oc rollout restart deployment nginx\n",
]


# ============================================================
# TOKENIZATION FUNCTION
# ============================================================
#
# WHAT IS TOKENIZATION?
# AI models don't understand text directly. They work with numbers.
# Tokenization converts text → numbers that the model can process.
#
# Example:
#   "show pods" → [1234, 5678]  (simplified)
#
# WHAT ARE LABELS?
# Labels tell the model "this is the correct answer".
# During training, the model tries to predict the labels.
# 
# We set padding tokens to -100 because:
# - Padding is just filler to make all examples the same length
# - We don't want the model to learn to predict padding
# - -100 is a special value that tells PyTorch "ignore this"
#
def tokenize_function(text: str, tokenizer) -> dict:
    """
    Convert text to tokens that the model can understand.
    
    Args:
        text: The training example (e.g., "User: list pods\nCommand: oc get pods")
        tokenizer: The tokenizer that knows how to convert text to numbers
    
    Returns:
        Dictionary with:
        - input_ids: The text converted to numbers
        - attention_mask: Which tokens are real (1) vs padding (0)
        - labels: What the model should predict (same as input_ids, but -100 for padding)
    """
    tokens = tokenizer(
        text,
        truncation=True,        # Cut off if too long
        padding="max_length",   # Pad with zeros if too short
        max_length=MAX_LENGTH
    )
    
    # Create labels: same as input_ids, but -100 for padding
    # -100 means "don't compute loss for this token"
    tokens["labels"] = [
        t if t != tokenizer.pad_token_id else -100 
        for t in tokens["input_ids"]
    ]
    
    return tokens


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================
def main():
    print("=" * 60)
    print("FunctionGemma Fine-Tuning for OpenShift Commands")
    print("=" * 60)
    
    # --------------------------------------------------------
    # STEP 0: Fix GPU Driver
    # --------------------------------------------------------
    # This MUST be called before importing torch.cuda functions
    # See the function definition above for why this is needed
    fix_gpu_driver()
    
    # --------------------------------------------------------
    # STEP 1: Check GPU Availability
    # --------------------------------------------------------
    # GPU makes training ~10x faster, but CPU works too!
    # 
    # WHY float32 INSTEAD OF float16?
    # float16 (half precision) uses less memory and is faster,
    # BUT it can cause numerical issues with small models.
    # I spent hours debugging because float16 made the model
    # output only <pad> tokens. float32 is more stable.
    #
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device_map = "cuda"
        dtype = torch.float32  # float32 for stability!
    else:
        print("⚠️ No GPU found, using CPU (training will be slower)")
        device_map = "auto"
        dtype = torch.float32
    
    # --------------------------------------------------------
    # STEP 2: Load Model and Tokenizer
    # --------------------------------------------------------
    # 
    # TOKENIZER: Converts text ↔ numbers
    #   "hello" → [123, 456] → "hello"
    #
    # MODEL: The actual AI "brain"
    #   Takes numbers in, produces probability of next token
    #
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
    # STEP 3: Configure LoRA
    # --------------------------------------------------------
    #
    # LoRA adds tiny "adapter" layers to specific parts of the model.
    # We only train these adapters, not the whole model.
    #
    # target_modules: Which layers to add adapters to
    #   - q_proj, k_proj, v_proj, o_proj are attention layers
    #   - These are where the model "pays attention" to input
    #   - We only use q_proj and v_proj to keep it simple
    #
    # Think of it like this:
    #   Original model: A → B → C → D → Output
    #   With LoRA:      A → B(+adapter) → C → D(+adapter) → Output
    #   We only train the adapters!
    #
    print("\n⚙️ Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=LORA_RANK,                          # Adapter size
        lora_alpha=LORA_ALPHA,                # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Where to add adapters
        lora_dropout=LORA_DROPOUT,            # Regularization
        bias="none",                          # Don't train bias terms
        task_type=TaskType.CAUSAL_LM          # We're doing text generation
    )
    
    # Wrap the model with LoRA adapters
    peft_model = get_peft_model(model, lora_config)
    
    # Count parameters
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    
    print(f"✅ LoRA configured!")
    print(f"   Total params:     {total:,}")
    print(f"   Trainable params: {trainable:,}")
    print(f"   Trainable %:      {100 * trainable / total:.2f}%")
    
    # --------------------------------------------------------
    # STEP 4: Prepare Training Data
    # --------------------------------------------------------
    #
    # We convert our list of strings into a HuggingFace Dataset,
    # then tokenize each example.
    #
    # The Dataset class is like a smart list that:
    # - Handles batching automatically
    # - Can shuffle data
    # - Works nicely with the Trainer
    #
    print("\n📊 Preparing training data...")
    
    # Create dataset from our training examples
    dataset = Dataset.from_list([{"text": t} for t in TRAINING_DATA])
    
    # Tokenize all examples
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x["text"], tokenizer),
        remove_columns=["text"]  # Remove the original text, keep only tokens
    )
    
    # Verify that labels aren't all -100 (a bug I hit!)
    sample_labels = [l for l in tokenized_dataset[0]["labels"] if l != -100]
    print(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
    print(f"   Sample has {len(sample_labels)} trainable tokens")
    
    # --------------------------------------------------------
    # STEP 5: Configure Training
    # --------------------------------------------------------
    #
    # TrainingArguments: All the settings for training
    # Trainer: The class that actually runs training
    #
    # KEY SETTINGS:
    # - num_train_epochs: How many passes through the data
    # - learning_rate: How fast to learn (lower = more stable)
    # - fp16=False: Use full precision (more stable)
    # - warmup_ratio: Start with tiny learning rate, ramp up
    #
    print("\n🎯 Configuring training...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,        # First 10% of training: gradually increase LR
        logging_steps=25,        # Print loss every 25 steps
        save_strategy="no",      # Don't save checkpoints (we save at end)
        fp16=False,              # Keep False for stability!
        report_to="none",        # Don't send metrics to wandb/etc
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
    # STEP 6: Train!
    # --------------------------------------------------------
    #
    # This is where the magic happens!
    # The trainer will:
    # 1. Loop through all training data (one epoch)
    # 2. For each batch:
    #    - Run the model to get predictions
    #    - Compare predictions to labels (compute loss)
    #    - Update model weights to reduce loss
    # 3. Repeat for NUM_EPOCHS
    #
    # Loss should go DOWN over time. If it goes up or stays flat,
    # something is wrong (check learning rate, data format, etc.)
    #
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    trainer.train()
    
    # Get final loss for reporting
    losses = [l['loss'] for l in trainer.state.log_history if 'loss' in l]
    print("-" * 60)
    print(f"✅ Training complete!")
    print(f"   Loss: {losses[0]:.2f} → {losses[-1]:.2f}")
    
    # --------------------------------------------------------
    # STEP 7: Save Model
    # --------------------------------------------------------
    #
    # We save:
    # - The LoRA adapter weights (small, ~1-3 MB)
    # - The tokenizer (so we can use it later)
    #
    # Note: We're saving the ADAPTER, not the full model!
    # To use it later, you load the base model + adapter.
    #
    print("\n💾 Saving model...")
    
    peft_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Model saved to {OUTPUT_DIR}")
    
    # --------------------------------------------------------
    # STEP 8: Test the Model
    # --------------------------------------------------------
    #
    # Let's see if it actually learned something!
    # 
    # We:
    # 1. Put the model in "eval" mode (turns off dropout)
    # 2. Give it a prompt
    # 3. Let it generate the completion
    # 4. See if it matches what we expect
    #
    print("\n🧪 Testing fine-tuned model...")
    print("-" * 60)
    
    test_prompts = [
        "list pods",
        "get services",
        "list deployments",
        "get nodes",
        "show namespaces",
    ]
    
    peft_model.eval()  # Turn off dropout for inference
    
    for prompt in test_prompts:
        # Format input like our training data
        input_text = f"User: {prompt}\nCommand:"
        inputs = tokenizer(input_text, return_tensors="pt").to(peft_model.device)
        
        # Generate completion
        with torch.no_grad():  # Don't compute gradients (faster)
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=30,           # Generate up to 30 new tokens
                do_sample=False,             # Greedy decoding (deterministic)
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode and extract the command
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        command = response.split("Command:")[-1].strip().split("\n")[0] if "Command:" in response else "?"
        
        print(f"📝 Input:  '{prompt}'")
        print(f"🤖 Output: {command}")
        print()
    
    print("-" * 60)
    print("🎉 All done!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print("\n📚 What's next?")
    print("   - Add more training examples for better accuracy")
    print("   - Try the model with different prompts")
    print("   - Build a web interface")
    print("   - Deploy as an API")


if __name__ == "__main__":
    main()
