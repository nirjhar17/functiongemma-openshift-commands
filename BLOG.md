# I Fine-Tuned Google's Tiny AI Model to Understand OpenShift Commands — Here's How

*A beginner-friendly guide to fine-tuning FunctionGemma on Red Hat OpenShift AI with GPU*

---

## 🤔 What Problem Was I Trying to Solve?

Imagine you're managing a Kubernetes or OpenShift cluster. You want to see all running applications, but you can't remember the exact command. Wouldn't it be nice to just type:

> "Show me all running pods"

And have an AI translate that into:

```bash
oc get pods
```

That's exactly what I built! And I did it using one of the smallest AI models available — **Google's FunctionGemma** with only **270 million parameters**.

---

## 🧠 What is FunctionGemma?

### Think of it Like This

Most AI chatbots are designed to **talk** with you. FunctionGemma is designed to **do things**.

| Regular AI | FunctionGemma |
|------------|---------------|
| "The weather is sunny!" | `call:weather{location:"NYC"}` |
| Human conversation | Machine-readable commands |

Google built it for things like:
- "Turn off the lights" → `smart_home{action:"lights_off"}`
- "Set timer for 5 mins" → `timer{minutes:5}`

**My idea:** Train it to understand OpenShift commands!

### Why 270 Million Parameters?

| Model | Size | Can Run On |
|-------|------|------------|
| GPT-4 | ~1.7 Trillion | Data centers only |
| Llama 3 | 8-70 Billion | Good GPUs |
| **FunctionGemma** | **270 Million** | Laptop/Phone! |

---

## 🎯 The Goal: Before vs After

### Before (Base Model)
```
Input: "list pods"
Output: list pods User: list pods User: list pods...  ← Just repeating!
```

### After (Fine-Tuned)
```
Input: "list pods"
Output: oc get pods  ← Correct!
```

---

## 🛠️ Setting Up on OpenShift AI

### Step 1: Create a Workbench

1. Log into OpenShift AI Dashboard
2. Create a Data Science Project called "fine-tune"
3. Create a Workbench:
   - **Image:** PyTorch | CUDA | Python 3.12
   - **Size:** Medium
   - **GPU:** 1x NVIDIA (optional but faster)

### Step 2: Install Libraries

```python
!pip install transformers>=4.51.0 peft datasets accelerate --quiet
```

### Step 3: Login to HuggingFace

Go to https://huggingface.co/google/functiongemma-270m-it and accept the license first!

```python
from huggingface_hub import login
login(token="your-hf-token")
```

---

## 🔧 The GPU Driver Fix

Here's something I learned the hard way. On OpenShift AI, the GPU might not work out of the box!

**The Error:**
```
Error 803: system has unsupported display driver / cuda driver combination
```

**The Fix:** Add this before importing PyTorch:

```python
import ctypes
ctypes.CDLL('/lib64/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")  # Now it works!
```

Why? The container has old CUDA libraries that conflict with the new GPU driver. This loads the correct one.

---

## 📥 Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/functiongemma-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32, not float16!
    device_map="cuda"
)

print(f"✅ Loaded {model.num_parameters():,} parameters")
# Output: ✅ Loaded 268,835,456 parameters
```

**Important:** Use `float32` not `float16`. I wasted hours debugging because float16 caused the model to output garbage!

---

## ⚡ LoRA: The Secret Sauce

Instead of training all 269 million parameters, I used **LoRA** to train just 0.27% of them!

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,                           # Small adapter size
    lora_alpha=16,                 # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to modify
    lora_dropout=0.1,              # Prevent overfitting
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model, lora_config)
```

**Result:**
```
Total params:     268,835,456
Trainable params: 368,640
Trainable %:      0.14%
```

### What is LoRA?

Imagine the model's brain has millions of connections. LoRA:
1. **Freezes** all the original connections
2. **Adds tiny adapters** on top
3. Only trains the adapters

It's like putting sticky notes on a textbook instead of rewriting it!

---

## 📊 Training Data

I created 52 examples of natural language → commands:

```python
training_data = [
    "User: show all pods\nCommand: oc get pods\n",
    "User: list pods\nCommand: oc get pods\n",
    "User: scale nginx to 5\nCommand: oc scale deployment nginx --replicas=5\n",
    "User: delete pod test\nCommand: oc delete pod test\n",
    "User: get logs nginx\nCommand: oc logs nginx\n",
    # ... 47 more examples
]
```

**Key insight:** Multiple variations of the same command help the model learn better!

---

## 🚀 Training

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./finetuned-functiongemma",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    learning_rate=5e-5,  # Low learning rate!
    fp16=False,          # Keep this False!
)

trainer = Trainer(model=peft_model, args=training_args, train_dataset=dataset)
trainer.train()
```

**Training Progress (on Tesla T4 GPU):**
```
Step   Loss
25     8.19
50     5.12
100    2.73
200    1.61
300    1.54
```

Loss went from 8.2 → 1.4 in about 2 minutes!

---

## 🧪 Results

| Input | Output | Correct? |
|-------|--------|----------|
| "list pods" | oc get pods | ✅ |
| "list deployments" | oc get deployments | ✅ |
| "get services" | oc get services | ✅ |
| "get nodes" | oc get nodes | ✅ |
| "scale nginx to 5" | oc describe deployment nginx | ❌ |
| "delete pod test" | oc namespace pod | ❌ |

**Accuracy: 33%** on test set

### What Worked Well
- Simple "get X" commands: pods, deployments, services, nodes ✅

### What Needs More Training
- Complex commands with arguments (scale, delete)
- Commands with multiple words

---

## 💡 Lessons Learned

### 1. Use float32, not float16
Float16 caused numerical instability. The model would output only `<pad>` tokens.

### 2. Low Learning Rate
Started with 1e-3, model exploded. 5e-5 worked well.

### 3. GPU Driver Fix is Real
Spent 2 hours on the CUDA 803 error. The ctypes fix saved me.

### 4. More Data = Better Results
52 examples got me 33% accuracy. More diverse examples would help.

### 5. Small Models Have Limits
270M parameters is great for simple patterns, but struggles with complex commands.

---

## 📁 Project Files

```
functiongemma-finetuning/
├── README.md              # Quick start guide
├── BLOG.md                # This article
├── ISSUES.md              # Problems I hit and fixes
├── finetune_functiongemma.py  # Complete training script
├── training_data.json     # Training examples
└── requirements.txt       # Dependencies
```

---

## 🔜 What's Next?

Ideas for improvement:
- Add 200+ training examples
- Try a larger model (Gemma 2B)
- Build a web UI
- Add kubectl support
- Deploy as an API

---

## 📚 Resources

- [Google FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma)
- [HuggingFace Model](https://huggingface.co/google/functiongemma-270m-it)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

---

## 🙏 Summary

I successfully:
- ✅ Fine-tuned FunctionGemma (270M) on OpenShift AI
- ✅ Used LoRA to train only 0.14% of parameters
- ✅ Got GPU working with the driver fix
- ✅ Achieved 33% accuracy on OpenShift commands
- ✅ Learned a LOT about LLM training pitfalls

The model isn't perfect, but it proves the concept works. With more data and training, this could become a useful CLI assistant!

---

*Check out the [ISSUES.md](./ISSUES.md) file for all the problems I hit and how I fixed them!*

---

**Tags:** #MachineLearning #AI #OpenShift #Kubernetes #Google #Gemma #FineTuning #LoRA #Tutorial #GPU
