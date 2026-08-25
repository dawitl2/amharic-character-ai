# Amharic Character Recognition — Handout 2

This handout covers the completion of Phase 11 and Phase 12, advancing the project to the point where the model performs its first complete training step.

---

# New Files

```text
src/
├── optimizer_demo.py
└── first_training_step.py
```

---

# `optimizer_demo.py`

This file introduces the concept of an optimizer and gradient descent.

We use Stochastic Gradient Descent (SGD) to adjust the model's weights:

```text
torch.optim.SGD
```

The script demonstrates:

1. **Zeroing Gradients**: Clearing old gradients using `optimizer.zero_grad()` before calculating new ones.
2. **Forward and Backward Pass**: Calculating loss and computing new gradients.
3. **Updating Weights**: Calling `optimizer.step()` to adjust the model's biases and weights.
4. **Verifying Learning**: Checking that the model's internal values changed and that the loss has successfully decreased after the update.

---

# `first_training_step.py`

This script combines all previous lessons into the first real training step using our actual dataset.

It performs the following:

```text
loads a batch of real Ethiopic images
↓
generates logits using the simple model
↓
calculates cross-entropy loss
↓
clears previous gradients
↓
performs backpropagation
↓
updates weights using SGD
↓
calculates predictions and batch accuracy
```

This represents one single step of learning. The model looks at a tiny batch of data and slightly corrects itself.

---

# What Happens Next?

Currently, the model only takes one step and stops.

In Phase 13, we will build a **Training Loop** to repeat this step many times across the entire dataset. Repeating this process for multiple "epochs" is what truly trains the model to recognize all characters accurately.
