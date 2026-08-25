# Amharic Character Recognition — Handout 3

This handout covers Phase 13 of the project: creating a Training Loop. We now move from taking a single step of learning to training the model over the entire dataset across multiple passes.

---

# New Files

```text
src/
└── training_loop.py
```

---

# What is an Epoch and an Iteration?

When training a neural network, we use two key terms to describe how the data is processed:

* **Iteration**: One update of the model's weights using a single batch of images. (What we did in Phase 12).
* **Epoch**: One complete pass over the *entire* training dataset. 

If our training dataset has 23 images and our batch size is 4, one epoch requires 6 iterations (batches) to complete.

---

# `training_loop.py`

This script implements the full training process. 

It loops through the data multiple times (epochs). Inside each epoch, it loops through every available batch of data (iterations).

The nested loop structure looks conceptually like this:

```text
Repeat for N epochs:
    Loop over all batches in training data:
        clear gradients
        forward pass
        calculate loss
        backward pass
        update weights
    
    Calculate average loss for the epoch
    Calculate total accuracy for the epoch
    Print results
```

# Expected Results

As the model completes more epochs, we expect it to gradually learn from its mistakes. The loss should decrease and the accuracy should increase.

For example, when running our simple model:

```text
Epoch 1
Loss: 11.36
Accuracy: 39%
...
Epoch 5
Loss: 4.82
Accuracy: 57%
...
Epoch 10
Loss: 0.00
Accuracy: 100%
```

The exact numbers change each time due to random initialization and shuffling, but the trend shows the model successfully learning to recognize the training images! 

---

# What Happens Next?

The model currently learns successfully, but we need to track this progress better. In Phase 14, we will introduce ways to record and observe this learning over time (e.g., plotting graphs). 

We also need to eventually check whether the model is truly *learning* the shapes or just *memorizing* the exact training images, which we will address during validation (Phase 15).
