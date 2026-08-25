# Amharic Character Recognition — Handout 8

This handout covers Phase 18 of the project: Error Analysis. Even when a model reaches high accuracy, it is critical to understand *what* it gets wrong when it makes mistakes.

---

# New Files

```text
src/
└── error_analysis.py

confusion_matrix.png (Generated automatically)
```

---

# Why Error Analysis?

A model with 95% accuracy sounds great, but what if all 5% of its errors are confusing `ሀ` (Ha) with `ለ` (Le) and it completely ignores other characters? 

We need to dive deeper than a single "overall accuracy" number.

# Confusion Matrix

The `error_analysis.py` script uses `scikit-learn` to generate a **Confusion Matrix** (`confusion_matrix.png`).

A Confusion Matrix is a grid that shows:
- The True Labels on one axis.
- The Predicted Labels on the other axis.

When you look at the matrix, the **diagonal line** from top-left to bottom-right represents correct predictions. Any numbers *outside* of the diagonal are errors. The row and column of an error instantly tell you which characters the model is confusing.

# Classification Report

The script also prints a detailed per-class report. Instead of one accuracy number, we see:
- How accurate it is specifically when predicting `ሀ`.
- How accurate it is specifically when predicting `ለ`.
- How accurate it is specifically when predicting `መ`.

Because our current dataset is synthetic and extremely small, the model achieves perfect 100% accuracy, meaning the confusion matrix is perfectly diagonal and there are zero incorrect examples to display. As we introduce more complex and realistic data, this tool will become vital.

---

# What Happens Next?

Our model works and we have analyzed it. In Phase 19, we will learn how to **Save the Model** so we don't have to retrain it every time we want to run a prediction!
