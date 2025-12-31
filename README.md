# FunctionGemma Fine-Tuning for OpenShift Commands

Fine-tune Google's FunctionGemma (270M parameters) to convert natural language into OpenShift/Kubernetes CLI commands.

## 🎯 What This Does

**Input:** "Show me all running pods"  
**Output:** `oc get pods`

## 📋 Prerequisites

- Red Hat OpenShift AI (or any Jupyter environment)
- HuggingFace account with Gemma license accepted
- Python 3.10+
- No GPU required (CPU works fine!)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install transformers peft datasets accelerate
```

### 2. Login to HuggingFace

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

### 3. Load Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "google/functiongemma-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)
```

### 4. Configure LoRA

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model, lora_config)
```

### 5. Train

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./finetuned-functiongemma",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
```

## 📊 Training Data Format

```python
training_data = [
    {"input": "show me all pods", "output": "get pods"},
    {"input": "scale frontend to 5", "output": "scale deployment frontend --replicas=5"},
    {"input": "create project demo", "output": "new-project demo"},
    # Add more examples...
]
```

## 📁 Project Structure

```
functiongemma-finetuning/
├── README.md                 # This file
├── BLOG.md                   # Medium blog post
├── finetune.ipynb           # Jupyter notebook (on OpenShift AI)
├── training_data.json       # Training examples
└── finetuned-functiongemma/ # Saved model (after training)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## 🔧 Supported Commands

| Natural Language | OpenShift Command |
|-----------------|-------------------|
| show me all pods | `get pods` |
| list all pods | `get pods` |
| get pods in all namespaces | `get pods -A` |
| show pods with IPs | `get pods -o wide` |
| delete pod nginx | `delete pod nginx` |
| show all deployments | `get deployments` |
| scale frontend to 5 | `scale deployment frontend --replicas=5` |
| restart api deployment | `rollout restart deployment api` |
| show all projects | `get projects` |
| create project demo | `new-project demo` |
| show all routes | `get routes` |
| who am I | `whoami` |
| show all namespaces | `get namespaces` |
| show all events | `get events` |
| exec bash in nginx | `exec -it nginx -- /bin/bash` |
| apply config file | `apply -f config.yaml` |

## 📈 Model Statistics

| Metric | Value |
|--------|-------|
| Base Model | google/functiongemma-270m-it |
| Total Parameters | 268,835,456 |
| Trainable Parameters (LoRA) | 737,280 |
| Trainable % | 0.27% |
| Training Time (CPU) | ~10-15 minutes |
| Training Time (GPU) | ~1 minute |
| Adapter Size | ~3 MB |

## 🧪 Testing

```python
# Test the fine-tuned model
test_prompts = ["show me all pods", "create project staging"]

for prompt in test_prompts:
    input_text = f"User: {prompt}\nCommand: oc"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = peft_model.generate(**inputs, max_new_tokens=32)
    print(tokenizer.decode(outputs[0]))
```

## 🔗 Resources

- [Google FunctionGemma Docs](https://ai.google.dev/gemma/docs/functiongemma)
- [HuggingFace Model](https://huggingface.co/google/functiongemma-270m-it)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

## 📝 License

This project uses Google's Gemma model which requires accepting the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## 🤝 Contributing

Feel free to:
- Add more training examples
- Improve the training configuration
- Add support for kubectl commands
- Build a web interface

## 📧 Contact

Created as part of learning AI/ML on Red Hat OpenShift AI.

