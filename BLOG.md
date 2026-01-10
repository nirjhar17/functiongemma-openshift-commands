# Fine-Tuning FunctionGemma for OpenShift Commands with LoRA on OpenShift AI

*A step-by-step guide to teaching Google's tiny AI model to understand Kubernetes/OpenShift CLI*

---

## What Are We Building?

Imagine talking to your computer like this:

> **You:** "Show me all running pods"  
> **AI:** `oc get pods`

We're building an AI that understands plain English and converts it into OpenShift commands.

---

## Google FunctionGemma: A Tiny but Special AI

### Not Your Typical Chatbot

Most AI models you've heard of (ChatGPT, Claude, Llama) are **chat models** – they're designed to have conversations with you. They're like a friend who loves to talk.

**FunctionGemma is different.** It's a **function-calling model** – designed to understand what you want and generate structured function calls. 

> **Important:** FunctionGemma does not execute functions itself. It outputs structured text (like `oc get pods`) that your application code can then parse and execute. The model acts as a bridge between natural language and software actions.

| Chat Model (GPT, Llama) | Function Model (FunctionGemma) |
|-------------------------|-------------------------------|
| "The weather looks nice today! Would you like me to tell you more about the forecast?" | `weather(location="NYC")` |
| Talks to you | **Does things for you** |

### Why Only 270 Million Parameters?

Here's the cool part – FunctionGemma is **tiny**:

| Model | Parameters | Can Run On |
|-------|------------|------------|
| GPT-4 | ~1,700,000 Million | Massive data centers |
| Llama 70B | 70,000 Million | Expensive GPUs |
| **FunctionGemma** | **270 Million** | **Your laptop!** |

Google designed it to be small enough to run on phones for things like:
- "Turn off the lights" → `smart_home(lights="off")`
- "Set a 5 minute timer" → `timer(minutes=5)`

**My idea:** What if we teach it OpenShift commands instead?

---

## How Does Fine-Tuning Work?

### The Problem

FunctionGemma already knows **how** to call functions. But it doesn't know **OpenShift commands**. It's like hiring someone who knows how to use a phone, but doesn't have anyone's number saved.

### The Solution: Fine-Tuning

Fine-tuning means teaching the model new knowledge while keeping what it already knows. We show it examples:

```
You say: "list all pods"        → Model should output: oc get pods
You say: "scale nginx to 5"     → Model should output: oc scale deployment nginx --replicas=5
You say: "delete pod broken"    → Model should output: oc delete pod broken
```

After seeing enough examples, the model learns the pattern!

---

## LoRA: The Secret to Cheap Fine-Tuning

### The Challenge

Normally, fine-tuning means updating **all 270 million parameters**. That needs:
- Expensive GPUs
- Hours of training
- Lots of memory

### The Solution: LoRA (Low-Rank Adaptation)

LoRA is a clever trick. Instead of changing the whole model, we:

1. **Freeze** all original parameters (don't touch them!)
2. **Add tiny adapter layers** (only 368,000 new parameters)
3. **Train only the adapters**

Think of it like this:

> **Without LoRA:** Rewriting an entire textbook to add one chapter  
> **With LoRA:** Adding sticky notes to the existing textbook

### The Numbers

```
Original model:     268,835,456 parameters (FROZEN - don't change)
LoRA adapters:          368,640 parameters (TRAINABLE - learn new stuff)
                        ─────────
Trainable:                 0.14%  ← We only train this tiny part!
```

**Result:** Training takes 2 minutes instead of hours!

---

## The Architecture (Simple Version)

Here's what happens when you ask "show all pods":

```
┌─────────────────────────────────────────────────────────────┐
│                    FunctionGemma Model                       │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐     ┌────────────┐  │
│  │   Tokenizer  │ ──►  │  Transformer │ ──► │  Output    │  │
│  │ "show pods"  │      │   Layers     │     │ "get pods" │  │
│  │  → [123,456] │      │  (+ LoRA!)   │     │            │  │
│  └──────────────┘      └──────────────┘     └────────────┘  │
│                              │                               │
│                    ┌─────────┴─────────┐                    │
│                    │  LoRA Adapters    │                    │
│                    │  (the new stuff   │                    │
│                    │   we trained!)    │                    │
│                    └───────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

**Step by step:**

1. **Tokenizer** converts your text into numbers the model understands
2. **Transformer layers** process the input (this is the "brain")
3. **LoRA adapters** add our OpenShift knowledge on top
4. **Output** is the predicted command

---

## Setting Up on OpenShift AI

### Step 1: Create a Workbench

I used **Red Hat OpenShift AI** – a platform that makes AI workloads easy on Kubernetes.

1. Log into OpenShift AI Dashboard
2. Create a Data Science Project: "fine-tune"
3. Create a Workbench:
   - **Image:** PyTorch | CUDA | Python 3.12
   - **Size:** Medium
   - **GPU:** 1x NVIDIA (optional but faster)

### Step 2: Install Libraries

```python
!pip install transformers>=4.51.0 peft datasets accelerate --quiet
```

| Library | What It Does |
|---------|--------------|
| `transformers` | Load AI models from HuggingFace |
| `peft` | LoRA and other efficient fine-tuning methods |
| `datasets` | Handle training data |
| `accelerate` | Speed up training |

### Step 3: Login to HuggingFace

You need to accept Google's license first at [huggingface.co/google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)

```python
from huggingface_hub import login
login(token="your-hf-token")
```

---

## The GPU Driver Fix

Here's something I learned the hard way. On OpenShift AI, the GPU might not work!

**The Error:**
```
Error 803: system has unsupported display driver / cuda driver combination
```

**Why?** The container has old CUDA libraries that conflict with the GPU driver.

**The Fix:** Load the correct driver BEFORE importing PyTorch:

```python
import ctypes
ctypes.CDLL('/lib64/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")  # Now it works!
```

---

## Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/functiongemma-270m-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for stability!
    device_map="cuda"
)

print(f"Loaded {model.num_parameters():,} parameters")
# Output: Loaded 268,835,456 parameters
```

**Important:** Use `float32` not `float16`. I wasted hours debugging because float16 made the model output garbage!

---

## Setting Up LoRA

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,                                    # Adapter size (small = fast)
    lora_alpha=16,                          # Scaling factor
    target_modules=["q_proj", "v_proj"],    # Which layers to adapt
    lora_dropout=0.1,                       # Prevent overfitting
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model, lora_config)
```

**What this does:**
- Freezes all 268 million original parameters
- Adds 368,640 trainable adapter parameters
- Now we only train 0.14% of the model!

---

## Training Data

I created 53 examples of natural language → commands:

```python
training_data = [
    "User: show all pods\nCommand: oc get pods\n",
    "User: list pods\nCommand: oc get pods\n",
    "User: scale nginx to 5\nCommand: oc scale deployment nginx --replicas=5\n",
    "User: delete pod test\nCommand: oc delete pod test\n",
    "User: get logs nginx\nCommand: oc logs nginx\n",
    # ... 48 more examples
]
```

**Tip:** Multiple variations of the same command help the model learn better!

---

## Training

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./finetuned-functiongemma",
    num_train_epochs=30,           # Go through data 30 times
    per_device_train_batch_size=4, # Process 4 examples at once
    learning_rate=5e-5,            # Small steps = stable learning
    fp16=False,                    # Keep False for stability!
)

trainer = Trainer(model=peft_model, args=training_args, train_dataset=dataset)
trainer.train()
```

**On Tesla T4 GPU:** Training took ~2 minutes!

**Training Progress:**
```
Step    Loss
25      8.19   ← Model is confused
50      5.12
100     2.73
200     1.61
300     1.44   ← Model is learning!
```

> **Note:** With 53 examples and batch size 4, each epoch has ~13 steps. Over 30 epochs, that's roughly 390 total training steps.

---

## Results

| Input | Output | Correct? |
|-------|--------|----------|
| "list pods" | oc get pods | Yes |
| "list deployments" | oc get deployments | Yes |
| "get services" | oc get services | Yes |
| "get nodes" | oc get nodes | Yes |
| "scale nginx to 5" | oc describe deployment nginx | No |

**Accuracy: 33%** with 53 training examples.

### What Worked
- Simple "get X" commands

### What Needs More Training
- Complex commands with arguments (scale, delete)
- More training examples would help!

---

## What I Learned

1. **Small models can be useful** – 270M parameters is enough for specific tasks
2. **LoRA makes fine-tuning accessible** – No expensive hardware needed
3. **Use float32, not float16** – Stability matters more than speed
4. **GPU driver issues are real** – The ctypes fix saved hours of debugging
5. **More data = better results** – 53 examples got 33% accuracy

---

## Project Files

All code is on GitHub: [functiongemma-openshift-commands](https://github.com/nirjhar17/functiongemma-openshift-commands)

```
├── README.md              # Quick start guide
├── BLOG.md                # This article
├── ISSUES.md              # Problems I hit and solutions
├── finetune_functiongemma.py  # Complete training script
└── training_data.json     # Training examples
```

---

## What's Next?

- Add 200+ training examples for better accuracy
- Try a larger model (Gemma 2B)
- Build a web interface
- Deploy as an API

---

## Resources

- [Google FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma)
- [HuggingFace Model](https://huggingface.co/google/functiongemma-270m-it)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

---

**Questions?** Check [ISSUES.md](./ISSUES.md) for common problems and solutions!

---

**Tags:** #MachineLearning #AI #OpenShift #Kubernetes #Google #Gemma #FineTuning #LoRA #Tutorial
