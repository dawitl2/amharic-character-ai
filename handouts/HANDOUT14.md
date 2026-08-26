# HANDOUT 14 — Understand Probabilities

In Phase 21, you successfully built an inference script and GUI. Now in Phase 22, we shift focus from predicting a single answer to understanding the model's confidence across all possible answers.

## Logits vs. Probabilities
Your neural network outputs "logits"—raw, unbounded numbers. For example:
`[2.5, -1.2, 0.4]`

These numbers don't tell us much intuitively. To make them useful, we pass them through a function called **Softmax**. 
Softmax does two things:
1. It turns all values into positive numbers (using exponents).
2. It scales them so that they all sum to `1.0` (or 100%).

Applying Softmax to the above logits might yield:
`[0.85, 0.02, 0.13]` 

Now we can say the model is 85% confident in the first class, 2% in the second, and 13% in the third.

## Why Softmax isn't in our Model architecture
You might notice that `SimpleModel` does not have a Softmax layer at the end. Why?
Because in PyTorch, `CrossEntropyLoss` automatically applies Softmax internally! Adding it to the model explicitly would cause it to be applied twice during training, which would ruin the learning process.

We only apply Softmax manually during **inference** (when predicting in `predict.py` and `gui.py`), where we actually want to see the percentages.

## A Warning on Confidence
Always remember: **Confidence is not correctness.** 
A model can be 99% confident and still be completely wrong. High confidence simply means the image strongly activated the mathematical patterns the model learned for that class. If the model learned bad patterns (or if the image is out-of-distribution), it will confidently make a mistake!
