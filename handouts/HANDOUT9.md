# Amharic Character Recognition — Handout 9

This handout covers Phase 19 of the project: Saving the Model. Training a neural network can take hours, days, or even weeks for large systems. We must save the results of that training so we can reuse it instantly.

---

# New Files

```text
src/
└── save_model.py

models/
├── simple_model_weights.pth
└── model_config.json
```

---

# What Are We Actually Saving?

A neural network's "knowledge" is stored entirely within its parameters (its **weights** and **biases**). 

In PyTorch, the collection of all these learned numbers is called the `state_dict` (State Dictionary). When we save a model, we are not saving the Python source code or the training dataset; we are just saving this dictionary of learned numbers to a `.pth` file.

# The Configuration File

Saving the weights is useless if we don't remember what they belong to. The model needs to know exactly what architecture was used to create those weights.

It also needs to remember the class mapping:
```json
{
    "ሀ": 0,
    "ለ": 1,
    "መ": 2
}
```
If we forget this mapping, the model might output `0` and we wouldn't know if that meant `ሀ` or `ለ`! 

We save this vital metadata into `model_config.json`.

---

# What Happens Next?

Now that our trained weights are safely saved on the hard drive, we can delete the dataset and stop the training loops. In Phase 20, we will learn how to **Load the Model** back into memory to use it.
