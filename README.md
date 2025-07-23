#  Vision Transformer (ViT) from Scratch

This repository contains a from-scratch implementation of a **Vision Transformer (ViT)** model in PyTorch, trained on the CIFAR-10 dataset. No pre-trained weights, no shortcuts — just pure transformer magic.

---

## 📦 Features

- ✅ Pure PyTorch implementation of ViT
- ✅ Patch embedding, positional encoding, multi-head attention, and MLP blocks
- ✅ LayerNorm, GELU, dropout regularization
- ✅ Configurable transformer depth, heads, and hidden sizes
- ✅ Training on CIFAR-10 with AdamW + Cosine Annealing LR
- ✅ Evaluation metrics: accuracy, loss, training/validation split

---

## 📊 Results

| Epochs | Training Acc | Validation Acc |
|--------|--------------|----------------|
| 20     | ~60%         | ~62%           |
| 50     | ~85%         | ~70% (Overfitting)          | 


📝 *Model architecture:*
- Image size: `32x32`
- Patch size: `4x4`
- Embedding dim: `192`
- Transformer layers: `6`
- Attention heads: `3`
- MLP hidden dim: `384`
- Dropout: `0.1`
---
The model with these params is overfitting on CIFAR-10 (small dataset) as it is only a base implementation.
However, tweaking  with better params and data augumentation can surely help.

---
## Requirements
```bash
pip install torch, torchvision, tqdm
 ```
## Clone this Repo
```bash
git clone https://github.com/rwtarpit/ViT-from-Scratch.git
cd ViT-from-Scratch
```


