# Amharic Character Recognition — Handout 10

This handout covers Phase 20 of the project: Loading the Model. We successfully take a trained model from our hard drive and bring it back to life to make real predictions!

---

# New Files

```text
src/
└── load_model.py
```

---

# Rebuilding the Brain

When we load a model, we are conceptually doing three distinct things:

### 1. Recreating the Architecture
We must recreate the exact same `SimpleModel` structure in memory. The architecture is the "shape" of the brain. If we change the architecture (e.g., adding another linear layer), the old weights will not fit.

### 2. Loading the Weights
We load `models/simple_model_weights.pth` and inject those learned numbers into our recreated `SimpleModel`. Now the empty brain has its memories back.

### 3. Setting Evaluation Mode
We MUST call `model.eval()`. This locks the model so its weights can't accidentally be changed during inference. 

---

# Real Inference
In the script, we manually load a `.png` file representing the character `ሀ`, transform it into a tensor, and pass it to the loaded model.

The model successfully outputs the correct prediction based on the training we performed in the previous phase, without having to run a single training loop!

---

# What Happens Next?

We have successfully completed a full end-to-end Machine Learning pipeline on a tiny "toy" dataset (Phase 1 through 20). 

However, 33 images are mathematically insignificant. In the next phase, we are going to discard our toy dataset, generate a **massive dataset** with thousands of images, and run the entire pipeline again to see how a real, scaled-up training process looks!
