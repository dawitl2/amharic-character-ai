# HANDOUT 16 — Deep Learning with CNNs & Experiment Tracking

## Phase 25: Building a Convolutional Neural Network (CNN)
Our original `LinearModel` flattened images into a 1D array of 4096 pixels, destroying spatial relationships. A **CNN (Convolutional Neural Network)** operates directly on 2D images. 

Key components introduced in `cnn_model.py`:
- **Conv2d (Convolutional Layers):** Learn 2D filters (kernels) that sweep across the image to detect edges, curves, and ultimately complex strokes.
- **MaxPool2d (Pooling Layers):** Downsample the image by taking the maximum value in a small window. This provides translation invariance (so the character can shift slightly) and reduces computational cost.
- **ReLU (Activation):** Introduces non-linearity so the network can learn complex patterns.

By using a CNN, the model can now recognize characters much more accurately, even with the heavy noise, blur, and shifts we introduced in Phase 23.

## Phase 26: Visualizing What the CNN Learns
Because CNNs maintain spatial structure, we can peek inside to see what they learn:
- **Early layers** (like `conv1`) typically learn basic edges and contrast detectors.
- **Deeper layers** (like `conv2`) combine those edges into specific loops, intersections, or strokes that make up Amharic characters.

## Phase 27: Proper Experiment Tracking
When training complex models on large datasets, you can't rely on memory. We've introduced `training_metrics.csv` to track our progress automatically. 
Every epoch, the pipeline records:
- Training Accuracy
- Validation Accuracy
- Training Loss

This allows us to track overfitting historically, plot learning curves, and ensure our changes to hyperparameters (like `LEARNING_RATE` or `BATCH_SIZE`) are actually improving the model.

### Legacy Support
The old linear model was renamed to `LinearModel` and kept as a reference in `linear_model.py`. Our main scripts (`train.py`, `predict.py`, `gui.py`) have fully migrated to the new `CNNModel`.
