# HANDOUT 15 — Improve Data Quality & Expand Classes

## Phase 23: Data Quality & Augmentation
Neural networks are only as good as the data they are trained on. Our early dataset used limited variations, which could allow the model to "memorize" specific pixel patterns rather than learning the actual shape of the characters (overfitting).

To force the model to learn robust features, we apply **Data Augmentation** during dataset generation:
- **Spatial variations:** Translations (shifting up/down/left/right), rotations, and scaling.
- **Image quality degradations:** Adding Gaussian blur, contrast/brightness adjustments, and random noise to simulate scanning or camera artifacts.
- **Background variations:** Not all documents are pure white. Introducing off-white and lightly textured backgrounds helps the model generalize.

By making the training task harder, the resulting model becomes more resilient to real-world messy data.

## Phase 24: Expanding Character Classes
So far, we've only recognized three characters (ሀ, ለ, መ). In Phase 24, we gradually expand our character set. We will bump it up to 10 base characters:
`ሀ, ለ, ሐ, መ, ሠ, ረ, ሰ, ሸ, ቀ, በ`

Expanding classes introduces new challenges:
- **Visually similar characters:** As the set grows, the model has to distinguish between characters that look very alike (e.g., ሰ and ሸ).
- **Dataset balance:** We must ensure we generate exactly the same number of examples for every class, so the model doesn't become biased toward the most common characters.
- **Model capacity:** A simple linear model might start to struggle as the problem complexity increases. This naturally leads us toward more advanced architectures in the future.
