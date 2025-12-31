# FunctionGemma Fine-Tuning for OpenShift Commands

Fine-tune Google's FunctionGemma (270M parameters) to convert natural language into OpenShift/Kubernetes CLI commands.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![GPU](https://img.shields.io/badge/GPU-Optional-orange)

## 🎯 What This Does

**Input:** "Show me all running pods"  
**Output:** `oc get pods`

## ✨ Features

- 🤖 Fine-tunes Google's FunctionGemma (270M params)
- ⚡ Uses LoRA - trains only 0.14% of parameters
- 🖥️ Works on CPU or GPU
- 📦 Tested on Red Hat OpenShift AI
- 🔧 Includes GPU driver fix for CUDA 13

## 📋 Prerequisites

- Python 3.10+
- HuggingFace account ([accept Gemma license](https://huggingface.co/google/functiongemma-270m-it))
- 8GB RAM minimum
- GPU optional (training takes ~2 min on GPU, ~15 min on CPU)

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/functiongemma-finetuning.git
cd functiongemma-finetuning
pip install -r requirements.txt
```

### 2. Login to HuggingFace

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

### 3. Run Training

```bash
python finetune_functiongemma.py
```

## 🔧 GPU Driver Fix (OpenShift AI)

If you get `Error 803: unsupported display driver`, add this before importing torch:

```python
import ctypes
ctypes.CDLL('/lib64/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)

import torch  # Now CUDA works!
```

## 📊 Results

| Command Type | Accuracy |
|--------------|----------|
| Get resources (pods, services, nodes) | ✅ High |
| Scale deployments | ⚠️ Medium |
| Complex commands | ❌ Needs more training |

**Overall Accuracy:** 33% (with 52 training examples)

## 📁 Project Structure

```
functiongemma-finetuning/
├── README.md                    # This file
├── BLOG.md                      # Detailed blog post
├── ISSUES.md                    # Problems and solutions
├── finetune_functiongemma.py    # Training script
├── training_data.json           # Training examples
└── requirements.txt             # Dependencies
```

## 🔧 Supported Commands

| Natural Language | OpenShift Command |
|-----------------|-------------------|
| list pods | `oc get pods` |
| show deployments | `oc get deployments` |
| get services | `oc get services` |
| get nodes | `oc get nodes` |
| show namespaces | `oc get namespaces` |
| get routes | `oc get routes` |
| get events | `oc get events` |
| get secrets | `oc get secrets` |

## 📈 Training Stats

| Metric | Value |
|--------|-------|
| Base Model | google/functiongemma-270m-it |
| Total Parameters | 268,835,456 |
| Trainable (LoRA) | 368,640 (0.14%) |
| Training Time (GPU) | ~2 minutes |
| Training Time (CPU) | ~15 minutes |
| Training Examples | 52 |

## ⚠️ Common Issues

See [ISSUES.md](./ISSUES.md) for solutions to:

- GPU driver mismatch (Error 803)
- Transformers version errors
- Model outputting pad tokens
- Loss going to 0

## 🧪 Testing

```python
peft_model.eval()

input_text = "User: list pods\nCommand:"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = peft_model.generate(**inputs, max_new_tokens=30, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: User: list pods
#         Command: oc get pods
```

## 🔗 Resources

- [Google FunctionGemma Docs](https://ai.google.dev/gemma/docs/functiongemma)
- [HuggingFace Model](https://huggingface.co/google/functiongemma-270m-it)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

## 📝 License

This project is for educational purposes. Uses Google's Gemma model which requires accepting the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## 🤝 Contributing

Contributions welcome! Ideas:
- Add more training examples
- Support kubectl commands
- Build a web interface
- Try larger models

## 👤 Author

**Nirjhar Jajodia**

Created while learning AI/ML on Red Hat OpenShift AI.

---

⭐ Star this repo if you found it helpful!
