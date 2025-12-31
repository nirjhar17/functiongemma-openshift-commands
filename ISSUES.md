# Issues I Faced (And How I Fixed Them)

A quick guide to problems you might hit when fine-tuning FunctionGemma on OpenShift AI.

---

## 🔴 Issue 1: GPU Driver Mismatch

**Error:**
```
CUDA initialization: Error 803: system has unsupported display driver / cuda driver combination
```

**Why:** The PyTorch image has CUDA 12.8, but the GPU node has a newer driver (580.x with CUDA 13.0). The container's compatibility libraries conflict with the actual driver.

**Fix:** Load the correct driver library before importing PyTorch:

```python
import ctypes
ctypes.CDLL('/lib64/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)

import torch  # Now CUDA works!
```

---

## 🔴 Issue 2: Transformers Version Too Old

**Error:**
```
KeyError: 'gemma3_text'
ValueError: model type 'gemma3_text' not recognized
```

**Why:** FunctionGemma uses the Gemma 3 architecture. Older transformers versions don't support it.

**Fix:**
```bash
pip install transformers>=4.51.0
```

---

## 🔴 Issue 3: Model Outputs Only Pad Tokens

**Symptom:** After training, model generates `<pad><pad><pad>` instead of commands.

**Why:** 
1. Learning rate too high (model weights explode)
2. Using float16 caused numerical instability
3. Labels were all masked (-100)

**Fix:**
- Use `torch_dtype=torch.float32` (not float16)
- Lower learning rate: `5e-5` instead of `1e-3`
- Set `fp16=False` in TrainingArguments
- Make sure labels aren't all -100

---

## 🔴 Issue 4: Loss Goes to 0 But Model Doesn't Work

**Symptom:** Training loss = 0.0000, but model outputs garbage.

**Why:** Labels were incorrectly masked. If all labels are -100, loss is 0 but model learns nothing.

**Fix:** Proper label masking - only mask padding tokens:

```python
tokens["labels"] = [
    t if t != tokenizer.pad_token_id else -100 
    for t in tokens["input_ids"]
]
```

---

## 🔴 Issue 5: PEFT/Transformers Version Conflict

**Error:**
```
ModuleNotFoundError: No module named 'transformers.modeling_layers'
ImportError: cannot import name 'MODEL_TYPE_TO_PEFT_MODEL_MAPPING'
```

**Why:** PEFT and transformers versions don't match.

**Fix:**
```bash
pip install transformers>=4.51.0 peft==0.13.0
```

Then **restart the kernel**.

---

## 🔴 Issue 6: GPU Not Available After Installing Packages

**Symptom:** `torch.cuda.is_available()` returns `False` after pip install.

**Why:** Installing packages can change PyTorch version or break CUDA.

**Fix:** 
1. Restart kernel after installing packages
2. Apply GPU driver fix (Issue 1) before importing torch

---

## 🔴 Issue 7: HuggingFace Token Error

**Error:**
```
HfHubHTTPError: 401 Unauthorized - Invalid user token
```

**Why:** Wrong token or token not set.

**Fix:**
1. Get token from https://huggingface.co/settings/tokens
2. Accept Gemma license at https://huggingface.co/google/functiongemma-270m-it
3. Use correct token:

```python
from huggingface_hub import login
login(token="hf_YOUR_ACTUAL_TOKEN")
```

---

## 🔴 Issue 8: Model Overfitting (Accuracy Gets Worse)

**Symptom:** Training longer makes results worse, not better.

**Why:** Small dataset (20-50 examples) + too many epochs = overfitting.

**Fix:**
- Add more training examples (50+ recommended)
- Reduce epochs (30-50 is enough)
- Lower learning rate
- Use dropout (lora_dropout=0.1)

---

## 🔴 Issue 9: Workbench Pod Stuck in Pending

**Error:** Pod status shows `FailedScheduling` - not enough GPU.

**Why:** Other models (like Qwen) are using all GPUs.

**Fix:**
1. Check what's using GPUs: `kubectl get pods -A | grep gpu`
2. Scale down unused models
3. Delete stuck pods

---

## 🔴 Issue 10: ImagePullBackOff

**Error:** Pod can't pull the PyTorch image.

**Why:** Internal registry issue or network problem.

**Fix:** Wait a few minutes and try again. The registry sometimes has temporary issues.

---

## ✅ Summary: My Recommended Setup

```python
# GPU Driver Fix
import ctypes
ctypes.CDLL('/lib64/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)

# Stable Settings
model = AutoModelForCausalLM.from_pretrained(
    "google/functiongemma-270m-it",
    torch_dtype=torch.float32,  # NOT float16
    device_map="cuda"
)

# Conservative LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Fewer modules
    lora_dropout=0.1
)

# Safe Training
TrainingArguments(
    num_train_epochs=30,
    learning_rate=5e-5,  # Low!
    fp16=False,  # Disabled
)
```

---

**Questions?** These are the issues I hit. If you find new ones, feel free to add them!

