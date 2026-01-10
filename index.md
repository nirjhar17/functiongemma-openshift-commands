# FunctionGemma Fine-Tuning Project

Welcome! This project demonstrates how to fine-tune Google's FunctionGemma (270M parameters) to understand OpenShift/Kubernetes commands.

## Read the Full Blog

**[Read the Complete Tutorial](./BLOG.md)** - Step-by-step guide with code

## Quick Links

- [Training Script](./finetune_functiongemma.py)
- [Training Data](./training_data.json)
- [Common Issues & Solutions](./ISSUES.md)
- [GitHub Repository](https://github.com/nirjhar17/functiongemma-openshift-commands)

## What This Project Does

Converts natural language to OpenShift commands:

| You Say | AI Outputs |
|---------|------------|
| "show all pods" | `oc get pods` |
| "list deployments" | `oc get deployments` |
| "get services" | `oc get services` |

