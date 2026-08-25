# Amharic Character Recognition — Handout 11 (The Scale Phase)

This handout covers our first independent milestone after completing the fundamental phases 1-20: **Scaling the Data**.

---

# New Files

```text
src/
├── generate_large_dataset.py
└── train_large_dataset.py

large_dataset_curve.png (Generated automatically)
```

---

# The Problem with Toy Datasets

Up until Phase 20, we used a dataset of exactly 33 images. While it helped us learn the fundamentals of PyTorch, it is practically useless for real-world OCR (Optical Character Recognition). The model reached 100% test accuracy only because every `ሀ` looked exactly like the others. 

If we showed our previous model a slightly blurry `ሀ`, or a `ሀ` that was shifted slightly to the left, it would have failed completely.

# `generate_large_dataset.py`

To build a robust model, we need realistic variation. This script throws away the 33 images and generates **6,000 images** (2,000 per class). 

We introduced deliberate randomness (Data Augmentation) during generation:
1. **Random Fonts**: Switching between Abyssinica SIL and Nyala.
2. **Random Scale**: Changing font sizes between 24 and 56.
3. **Random Shifts**: Moving the character up, down, left, and right randomly.
4. **Random Rotation**: Tilting the characters up to 15 degrees.
5. **Blur & Noise**: Adding blur and static noise to simulate bad camera quality or scanning artifacts.

# `train_large_dataset.py`

We then ran our exact same model architecture (`SimpleModel` from Phase 7) on this massive, messy dataset.

**Changes made for scale:**
- **Batch Size**: We increased the batch size from 4 to 64. Passing 64 images at a time allows the GPU/CPU to calculate gradients much faster.
- **Robust Evaluation**: Our Train, Validation, and Test splits are now based on thousands of images. When the model scores high accuracy now, we can statistically trust that it actually learned the shapes, rather than just memorizing 33 images.

---

# Did It Get Better?

Yes! Even though the task is now much, much harder due to rotations, shifts, and noise, our simple network architecture proved capable of learning the general shapes of the characters. We successfully trained a model on a significantly larger and more complex dataset, proving that our fundamentals are sound.

As we add more character classes (Phase 24), we will eventually hit a ceiling where this `SimpleModel` isn't powerful enough, which will lead us into Convolutional Neural Networks (Phase 25)!
