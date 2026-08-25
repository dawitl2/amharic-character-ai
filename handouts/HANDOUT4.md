# Amharic Character Recognition — Handout 4

This handout covers Phase 14 of the project: Observing Learning. Tracking numbers in a terminal is difficult to read. In this phase, we introduce visualization to truly observe the model's learning process.

---

# New Files

```text
src/
└── observe_learning.py

learning_curve.png (Generated automatically)
```

---

# `observe_learning.py`

This script takes the training loop we built in Phase 13 and adds recording and plotting capabilities. 

We use Python's `matplotlib` library to observe two critical metrics:

### 1. Training Loss
Loss is the ultimate measure of model error. A successful learning process will show the loss curving downward over time (epochs). If the learning rate is too high or the training is unstable, this curve might zig-zag wildly.

### 2. Training Accuracy
Accuracy represents the percentage of images the model correctly guesses in the training batch. A successful training process will show this metric curving upward and eventually stabilizing near 100%.

---

# Convergence

When observing the graphs in `learning_curve.png`, you will notice that the improvements eventually slow down and flatten out. This flattening point is known as **convergence**. 

At convergence, the model has learned as much as it easily can from this specific training dataset under its current settings. Running for 1,000 more epochs once convergence is reached generally provides no benefit and just wastes compute time.

---

# What Happens Next?

Right now, we are evaluating the model solely on the exact same data it was trained on. It is easy to score 100% on a test when you've already seen all the answers!

In Phase 15, we will introduce **Validation**. We will pause training periodically and test the model on images it has *never* seen before to check if it has truly generalized its understanding of Amharic characters.
