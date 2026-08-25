# Amharic Character Recognition — Handout 7

This handout covers Phase 17 of the project: Final Test Evaluation. 

We have successfully trained the model, validated it, and checked for overfitting. Now, we put it to the ultimate test.

---

# New Files

```text
src/
└── test_evaluation.py
```

---

# The Golden Rule of Machine Learning

During development, you will change your model's architecture, adjust the learning rate, change the number of epochs, and add more data. You make these decisions based on how well the model performs on the **Validation Set**.

Because you are using the validation score to make decisions, you are slowly (and accidentally) tuning the model to perfectly fit the validation set. 

This introduces a massive risk: **Your model might look great on validation data, but still fail in the real world.**

To solve this, we strictly lock away a **Test Set** at the very beginning of the project.

**The Golden Rule**: The Test Set is evaluated *only once*, at the very end of the project, when all development decisions are frozen.

---

# `test_evaluation.py`

In this script, we perform the full pipeline:
1. Split the data into Train (70%), Validation (15%), and Test (15%).
2. Train the model for 20 epochs using the Train set.
3. Validate it periodically using the Validation set.
4. Freeze all decisions.
5. Evaluate it **one single time** on the Test set.

Our final test accuracy reached **100%**! 

*(Note: 100% is only possible right now because our dataset is tiny and contains clean, synthetic, identical font images. Real-world handwritten data will be much harder!)*

---

# What Happens Next?

The model is trained and successfully tested! We know it works. 

However, when the model makes a mistake, how do we know *why*? In Phase 18, we will explore **Error Analysis** to figure out which Amharic characters the model frequently confuses with each other.
