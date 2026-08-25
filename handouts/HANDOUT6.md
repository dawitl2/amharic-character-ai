# Amharic Character Recognition — Handout 6

This handout covers Phase 16 of the project: Overfitting and Underfitting. Now that we track validation accuracy, we can diagnose how well the model is learning.

---

# New Files

```text
src/
└── overfitting_demo.py

overfitting_curve.png (Generated automatically)
```

---

# Overfitting

Overfitting occurs when a model performs extremely well on its training data but terribly on unseen validation data. It has memorized the answers rather than learning the patterns.

In `overfitting_demo.py`, we intentionally caused the model to overfit by giving it only 6 images to learn from (a tiny training set), while testing it on a much larger validation set.

If you look at the generated `overfitting_curve.png`, you will see:
- **Training Accuracy (Blue)**: Quickly shoots up to 100%. The model easily memorized the 6 images.
- **Validation Accuracy (Red)**: Stays low and fluctuates wildly (often around 30-50%). The model fails to recognize the characters in the validation images because it never actually learned what makes a `ሀ` a `ሀ`.

### How to Fix Overfitting:
1. **More Data**: Expanding the dataset is the most effective solution.
2. **Data Augmentation**: Artificially create more data by slightly rotating, blurring, or moving the existing images.
3. **Regularization / Dropout**: Force the network to forget some things during training so it doesn't memorize exactly.
4. **Reduce Model Complexity**: A smaller model has less memory capacity, forcing it to learn general patterns instead of exact pixels.

---

# Underfitting

Underfitting is the opposite problem: the model performs poorly on *both* the training and validation sets. 

It means the model is not powerful enough to learn the patterns, or it hasn't been trained for enough epochs, or the learning rate is completely wrong.

---

# What Happens Next?

During development, you constantly tweak the model and settings, using the Validation accuracy to guide you. However, you might accidentally "overfit to the validation set" by choosing the exact settings that happen to score highest on those specific validation images.

To ensure our model is truly objective, Phase 17 introduces the **Final Test Evaluation**. We will freeze our decisions and evaluate the model one last time on data it has absolutely never seen before.
