# I Fine-Tuned Google's Tiny AI Model to Understand OpenShift Commands — Here's How

*A beginner-friendly guide to fine-tuning FunctionGemma on Red Hat OpenShift AI*

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

## 🧠 What is FunctionGemma? (Explained Simply)

### First, What is Gemma?

**Gemma** is Google's family of open-source AI models. Think of it as Google sharing their AI technology with the world for free. It's like Google saying, "Here's a smaller version of the AI brain we use — go build cool things with it!"

### What Makes FunctionGemma Special?

Most AI models are designed to **chat** with you — they generate human-like responses. But **FunctionGemma** is different. It's designed to **call functions**.

Think of it this way:

| Regular AI | FunctionGemma |
|------------|---------------|
| "The weather is sunny today!" | `call:weather{location:"New York"}` |
| Human conversation | Machine-readable commands |

Google originally built FunctionGemma for things like:
- "Turn off the lights" → `call:smart_home{action:"lights_off"}`
- "Set a timer for 5 minutes" → `call:timer{minutes:5}`

**My idea:** What if I train it to understand OpenShift/Kubernetes commands instead?

### Why Only 270 Million Parameters?

To put this in perspective:

| Model | Parameters | Size |
|-------|------------|------|
| GPT-4 | ~1.7 Trillion | Huge |
| Llama 3 | 8-70 Billion | Large |
| **FunctionGemma** | **270 Million** | **Tiny!** |

FunctionGemma is so small it can:
- ✅ Run on a CPU (no expensive GPU needed)
- ✅ Fit on a smartphone
- ✅ Train in minutes, not hours
- ✅ Deploy anywhere

---

## 🎯 The Goal: Before vs After Fine-Tuning

### Before Fine-Tuning (Base Model)

When I asked the base FunctionGemma model "Show me all pods", it had no idea what OpenShift was:

```
Input: "Show me all pods"
Output: call:oc{command:list pods}  ← Wrong! Not a real command
```

### After Fine-Tuning

After training on my custom dataset:

```
Input: "Show me all pods"
Output: call:oc{command:get pods}  ← Correct!
```

---

## 🛠️ Setting Up the Environment

### Step 1: Create an OpenShift AI Workbench

I used **Red Hat OpenShift AI** — a platform that makes running AI workloads on Kubernetes easy.

1. Log into OpenShift AI Dashboard
2. Create a new **Data Science Project** called "fine-tune"
3. Create a **Workbench** with:
   - Image: `PyTorch | CUDA | Python 3.12`
   - Container Size: Medium
   - No GPU needed (CPU works fine!)

### Step 2: Install Required Libraries

In my Jupyter notebook, I ran:

```python
!pip install transformers peft datasets accelerate --quiet
```

**What are these?**

| Library | Purpose |
|---------|---------|
| `transformers` | HuggingFace library to load AI models |
| `peft` | Parameter-Efficient Fine-Tuning (LoRA) |
| `datasets` | Handle training data |
| `accelerate` | Speed up training |

### Step 3: Login to HuggingFace

FunctionGemma requires accepting Google's license on HuggingFace:

```python
from huggingface_hub import login
login(token="your-huggingface-token")
```

---

## 📥 Loading the Model

### Step 4: Load FunctionGemma 270M

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/functiongemma-270m-it"

# Load tokenizer (converts text to numbers)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model (the AI brain)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)

print(f"✅ Model loaded! Parameters: {model.num_parameters():,}")
```

**Output:**
```
✅ Model loaded! Parameters: 268,835,456
```

That's 269 million parameters — tiny by modern standards!

---

## 🧪 Testing the Base Model

### Step 5: See What the Model Does Before Training

I created a function schema (telling the model what functions it can call):

```python
oc_function_schema = {
    "type": "function",
    "function": {
        "name": "oc",
        "description": "Execute an OpenShift CLI command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The oc command to execute"
                }
            },
            "required": ["command"]
        }
    }
}
```

Then tested it:

```python
messages = [
    {"role": "user", "content": "Show me all running pods"}
]

# The model generates a function call
# Output: call:oc{command:list pods}  ← Not quite right!
```

The base model knows it should call the `oc` function, but it doesn't know the correct OpenShift commands. That's what fine-tuning will fix!

---

## 📊 Creating the Training Dataset

### Step 6: Build Examples of What We Want

I created pairs of natural language → commands:

```python
training_data = [
    # Pod Operations
    {"input": "show me all pods", "output": "get pods"},
    {"input": "list all pods", "output": "get pods"},
    {"input": "get all pods in all namespaces", "output": "get pods -A"},
    {"input": "delete the nginx pod", "output": "delete pod nginx"},
    
    # Deployment Operations
    {"input": "show all deployments", "output": "get deployments"},
    {"input": "scale frontend to 5 replicas", "output": "scale deployment frontend --replicas=5"},
    {"input": "restart the api deployment", "output": "rollout restart deployment api"},
    
    # OpenShift Specific
    {"input": "show all projects", "output": "get projects"},
    {"input": "create new project demo", "output": "new-project demo"},
    {"input": "show all routes", "output": "get routes"},
    
    # And 60+ more examples...
]
```

**Key insight:** The more diverse examples you provide, the better the model learns!

---

## ⚡ Configuring LoRA (The Secret Sauce)

### Step 7: Make Training Efficient with LoRA

Here's the magic: Instead of training all 269 million parameters, I used **LoRA (Low-Rank Adaptation)** to train only **0.27%** of them!

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,                    # Rank (size of adapter)
    lora_alpha=16,          # Scaling factor
    target_modules=[        # Which layers to modify
        "q_proj", "k_proj", 
        "v_proj", "o_proj"
    ],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model, lora_config)
```

**Result:**
```
📊 Parameter Statistics:
   Total parameters:     268,835,456
   Trainable parameters: 737,280
   Trainable %:          0.27%
```

### What is LoRA? (Simple Explanation)

Imagine the model's brain has millions of connections. Instead of changing ALL connections (expensive!), LoRA:

1. **Freezes** the original connections
2. **Adds tiny "adapter" connections** on top
3. Only trains the adapters

It's like putting a small sticky note on a textbook instead of rewriting the whole book!

**Benefits:**
- ✅ 100x faster training
- ✅ 100x less memory
- ✅ Works on CPU
- ✅ Tiny adapter file (~3MB)

---

## 🚀 Training the Model

### Step 8: Let it Learn!

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./finetuned-functiongemma",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=5,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
```

**Training Progress:**
```
[1/6] Epoch 0.5/3
[2/6] Epoch 1.0/3
[3/6] Epoch 1.5/3
...
✅ Training complete!
```

On CPU, this took about **10-15 minutes** for 73 examples over 3 epochs. On GPU, it would be under 1 minute!

---

## 🔑 Key Takeaways

### What I Learned

1. **Small models can be powerful** — FunctionGemma at 270M parameters is perfect for specific tasks
2. **LoRA makes fine-tuning accessible** — No expensive hardware needed
3. **OpenShift AI is great for ML** — Easy to set up, Kubernetes-native
4. **Function calling models are different** — They output structured commands, not chat responses

### Why This Matters

This approach can be applied to:
- 🏠 Smart home control ("Turn off living room lights")
- 💻 CLI assistance ("How do I find large files?")
- 🤖 Automation ("Deploy the staging environment")
- 📊 Data queries ("Show me sales from last month")

---

## 🔜 What's Next?

In Part 2, I'll cover:
- Testing the fine-tuned model
- Saving and deploying the model
- Building a simple chat interface
- Comparing before vs after results

---

## 📚 Resources

- [Google FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma)
- [HuggingFace FunctionGemma](https://huggingface.co/google/functiongemma-270m-it)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

---

*Follow me for Part 2 where we test the model and build a demo!*

---

**Tags:** #MachineLearning #AI #OpenShift #Kubernetes #Google #Gemma #FineTuning #LoRA #Tutorial

