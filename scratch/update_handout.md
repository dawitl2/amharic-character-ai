
# Phase 28: Full Printed Character Expansion

We have expanded the dataset and model to support 290 standard Amharic characters (bases + orders + labiovelars + labialized forms). The dataset generates 1200 instances for each character using 4 different Amharic fonts, totaling 348,000 images.

Legacy linear models have been organized to `src/legacy_linear/`, and the GUI has been completely reimagined into a clean Tabview-based Light-theme interface to manage the increasingly complex diagnostic metrics.

Since the number of characters expanded from 10 to 290, old 10-class checkpoints are now strictly incompatible and safely rejected by the training system. We will start fresh retraining from random weights using the 290-class dataset.
