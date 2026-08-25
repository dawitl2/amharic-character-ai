# Amharic Character Recognition — Handout 5

This handout covers Phase 15 of the project: Validation. Until now, the model has been evaluated on the same data it uses to learn. We now introduce a "Validation Set" to test if the model is genuinely learning the character shapes.

---

# New Files

```text
src/
└── validation.py
```

---

# The Problem: Memorization vs. Generalization

If a student receives a practice test and memoizes the exact answers, scoring 100% does not mean they learned the subject. The same applies to neural networks.

If the model memorizes our training dataset perfectly (100% training accuracy) but fails when shown a new version of `ሀ` that it has never seen, it has failed to **generalize**.

# The Solution: Validation Split

In `validation.py`, we further split the dataset. Our total data is now roughly:
- **70% Training**: Used for backpropagation and updating weights.
- **15% Validation**: Never used to update weights. Used at the end of each epoch to test the model.
- **15% Test**: Locked away for final evaluation later.

---

# `model.train()` vs `model.eval()`

In `validation.py`, you will notice we switch the model's mode:

1. **`model.train()`**: Tells PyTorch we are actively learning.
2. **`model.eval()`**: Tells PyTorch to turn off training behaviors. We only want to use the model, not change it.

# `torch.no_grad()`

During the validation phase, we also use:
```python
with torch.no_grad():
```
This entirely disables gradient tracking. Since we do not perform backpropagation on the validation set, tracking gradients is a waste of computer memory and processing power.

---

# What Happens Next?

With training and validation accuracy side-by-side, we can now detect if the model is memorizing the data. When the training accuracy is extremely high (e.g., 100%) but the validation accuracy is very low, the model is **overfitting**. 

We will explore identifying and addressing overfitting in Phase 16.
