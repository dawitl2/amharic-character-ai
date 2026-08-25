# Amharic Character Recognition — Handout 12

This handout covers Phase 21 of the project: Real Inference. This is the stage where the model transforms from an isolated experiment into a functional, usable application!

---

# New Files

```text
src/
└── predict.py
```

---

# Real Inference (`predict.py`)

Previously, we only ever ran our model on batches of data handled by PyTorch's `DataLoader`. But in a real-world scenario (like a web app or mobile app), a user will upload a single raw image file.

The `predict.py` script acts as our application interface. It explicitly performs the following steps:

1. **Loads the Configuration & Weights**: Before doing anything, it loads the model architecture, the class mappings from `model_config.json`, and the saved trained weights from `simple_model_weights.pth`.
2. **Accepts an Image Path**: It accepts any image file path from the command line (e.g., `python src/predict.py data/ሀ/synthetic_001.png`).
3. **Preprocesses the Image**: The model expects a very specific mathematical format. The script uses the exact same `transforms` we used during training to:
   - Convert the image to Grayscale.
   - Resize it to exactly 64x64.
   - Convert it to a PyTorch Tensor.
   - Add a "Batch Dimension" (so `[1, 64, 64]` becomes `[1, 1, 64, 64]`, making the model think it's receiving a batch of size 1).
4. **Runs the Model**: It passes the image through the locked (`model.eval()`) model to obtain the raw **Logits** (scores).
5. **Calculates Probability**: Logits are raw, unbound numbers. The script applies the **Softmax** function (`F.softmax`) to mathematically crush these logits into probabilities between 0% and 100%.
6. **Translates the Prediction**: It finds the highest probability, retrieves the corresponding class index (e.g., `0`), and reverses it using the JSON configuration back into the human-readable Amharic character (e.g., `ሀ`).
7. **Displays the Result**: It outputs the predicted character alongside how confident the model is in its decision.

### Example Output:
```text
Prediction: ሀ
Confidence: 99.7%
```

---

# What Happens Next?

Our model can now be used on any command line to classify individual images. As we proceed through the next phases, we will explore expanding the character set and eventually moving to Convolutional Neural Networks (CNNs) to make our predictions incredibly robust even on messy, real-world, handwritten Amharic text!
