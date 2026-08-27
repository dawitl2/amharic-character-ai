# Amharic Character Recognition — Handout

This file is a simple explanation of the project and should be updated as the project progresses.

The purpose of the handout is to answer:

```text
What have we built?
What does each part mean?
Where are we right now?
What happens next?
```

---

# What Are We Building?

We are building a supervised-learning system that recognizes Amharic/Ethiopic characters.

Current goal:

```text
character image
↓
neural network
↓
predicted character
```

Current characters:

```text
ሀ
ለ
መ
```

PyTorch represents them as:

```text
ሀ → 0
ለ → 1
መ → 2
```

---

# Project Structure

```text
amharic-character-ai/
│
├── .venv/
├── .gitignore
├── data/
│   ├── ሀ/
│   ├── ለ/
│   └── መ/
│
└── src/
    ├── check_setup.py
    ├── generate_character.py
    ├── generate_dataset.py
    ├── inspect_image.py
    ├── inspect_dataset.py
    ├── split_dataset.py
    ├── inspect_dataloader.py
    ├── simple_model.py
    └── loss_demo.py
```

---

# `.venv`

Contains the Python libraries used by the project.

Examples:

```text
PyTorch
torchvision
NumPy
Pillow
Matplotlib
scikit-learn
```

It is ignored by Git and is not uploaded to GitHub.

---

# `data/`

Contains the images used by the model.

Current structure:

```text
data/
├── ሀ/
├── ለ/
└── መ/
```

The folder name is the correct answer.

Example:

```text
data/ሀ/synthetic_001.png
```

means:

```text
image = input
ሀ = correct label
```

This is why the project is supervised learning.

---

# `generate_character.py`

Creates one synthetic character image.

It:

```text
creates a 64 × 64 canvas
↓
loads an Ethiopic font
↓
draws a character
↓
centers it
↓
saves the image
```

---

# `generate_dataset.py`

Creates multiple versions of the characters.

Current dataset:

```text
ሀ
ለ
መ
```

using multiple fonts and font sizes.

We currently have about:

```text
33 images
```

This dataset is intentionally small while we learn the complete pipeline.

---

# `inspect_image.py`

Shows how an image becomes numbers.

A 64 × 64 grayscale image becomes:

```text
torch.Size([1, 64, 64])
```

Meaning:

```text
1 grayscale channel
64 height
64 width
```

The model does not literally see `ሀ`.

It sees numerical pixel values.

Approximately:

```text
0.0 = black
1.0 = white
```

---

# `inspect_dataset.py`

Loads the character folders as a PyTorch dataset.

It discovers:

```text
ሀ → 0
ለ → 1
መ → 2
```

Every training sample therefore contains:

```text
image tensor
+
correct numerical label
```

---

# `split_dataset.py`

Divides the data into:

```text
23 training images
5 validation images
5 test images
```

Training data teaches the model.

Validation data helps us evaluate it while developing.

Test data is used as a final examination.

---

# `inspect_dataloader.py`

Introduced batches.

Instead of sending one image at a time, we can send several together.

Example:

```text
torch.Size([4, 1, 64, 64])
```

means:

```text
4 images
1 grayscale channel
64 × 64 pixels
```

Their labels might look like:

```text
[1, 2, 0, 2]
```

which means:

```text
ለ
መ
ሀ
መ
```

---

# `simple_model.py`

Contains our first neural network.

It is deliberately simple:

```text
64 × 64 image
↓
4096 pixels
↓
Flatten
↓
Linear layer
↓
3 scores
```

The three scores correspond to:

```text
ሀ
ለ
መ
```

The model currently starts with random weights.

---

# Logits

The three raw scores produced by the model are called **logits**.

Example:

```text
[0.3, 1.8, 0.1]
```

means approximately:

```text
ሀ → 0.3
ለ → 1.8
መ → 0.1
```

The largest score gives the current prediction.

Here:

```text
1.8
```

is largest, so the model predicts:

```text
ለ
```

Logits are not probabilities.

They are raw model scores.

---

# Correct Label

The correct answer is only one number.

Example:

```text
1
```

means:

```text
ለ
```

We are **not** comparing three predefined numbers against three model numbers.

Instead:

```text
model produces three logits
+
dataset provides one correct label
↓
loss function
```

---

# Loss

Loss tells us how wrong the model currently is.

We use:

```text
CrossEntropyLoss
```

Conceptually:

```text
model outputs
+
correct answer
↓
loss
```

Lower loss generally means better predictions.

A perfect model would attempt to push loss toward zero.

---

# `loss_demo.py`

This is where we are currently working.

We manually created example logits and correct labels.

Then we calculated:

```text
loss = 0.3144
```

After that we ran:

```python
loss.backward()
```

This performs backpropagation.

---

# Gradient

A gradient tells us how changing a value would affect the loss.

Very simply:

```text
change this upward
change this downward
change this a lot
change this only slightly
```

Gradients provide directions for improving the model.

But gradients themselves do **not** change the model.

---

# Current Exact Position

```text
image
↓
model
↓
logits
↓
prediction
↓
correct label
↓
loss
↓
backpropagation
↓
gradients
↓
WE ARE HERE
↓
optimizer
↓
update weights
↓
repeat
↓
learning
```

The model has **not yet completed real training**.

We have learned how to calculate the information required for training.

The next major topic is the optimizer.

---

# What Happens Next?

The optimizer will use the gradients to modify the model's weights.

Then we can create the complete training cycle:

```text
images
↓
model
↓
predictions
↓
loss
↓
backpropagation
↓
gradients
↓
optimizer
↓
updated weights
↓
repeat
```

Repeating this process across the training dataset is what will actually teach the model to recognize the characters.

---

# Long-Term Direction

The project can eventually progress through:

```text
single printed characters
↓
larger Ethiopic character set
↓
CNN
↓
better datasets
↓
handwriting recognition
↓
character segmentation
↓
word recognition
↓
line recognition
↓
full Amharic OCR
↓
OCR + NLP
↓
context-aware Amharic document understanding
```

The handout should be updated whenever a major project milestone changes our understanding of the system.

# Phase 28: Full Printed Character Expansion

We have expanded the dataset and model to support 290 standard Amharic characters (bases + orders + labiovelars + labialized forms). The dataset generates 1200 instances for each character using 4 different Amharic fonts, totaling 348,000 images.

Legacy linear models have been organized to `src/legacy_linear/`, and the GUI has been completely reimagined into a clean Tabview-based Light-theme interface to manage the increasingly complex diagnostic metrics.

Since the number of characters expanded from 10 to 290, old 10-class checkpoints are now strictly incompatible and safely rejected by the training system. We will start fresh retraining from random weights using the 290-class dataset.
