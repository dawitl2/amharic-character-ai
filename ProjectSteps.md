# Amharic Character Recognition — Project Steps

This file is the long-term roadmap for building and understanding the project.

It should be updated as the system evolves.

The goal is not to rush through these steps.

Each stage should be understood before moving to the next one.

## Current milestone — August 2026

```text
290-class printed CharacterCNN
→ best validation accuracy 93.9751% at epoch 27
→ shared external-image preprocessing
→ OpenCV word/line segmentation
→ batched CNN character recognition
→ traceable word and sentence reconstruction
→ optional Amharic-to-English translation
→ professional desktop application and real training graphs
```

The interrupted training run is resumable from epoch 27. The full independent
test accuracy remains pending because the run did not reach final evaluation.

---

# PHASE 0 — Project Foundation

* [x] Decide the project goal.
* [x] Choose supervised learning.
* [x] Choose Python.
* [x] Choose PyTorch.
* [x] Create the project folder.
* [x] Open the project in VS Code.
* [x] Open PowerShell through VS Code.
* [x] Verify Python installation.
* [x] Create `.venv`.
* [x] Activate `.venv`.
* [x] Verify pip.
* [x] Install PyTorch.
* [x] Verify PyTorch.
* [x] Install NumPy.
* [x] Install torchvision.
* [x] Install Pillow.
* [x] Install Matplotlib.
* [x] Install scikit-learn.
* [x] Create `.gitignore`.
* [x] Ignore `.venv`.
* [x] Initialize Git.
* [x] Create private GitHub repository.
* [x] Connect local repository to GitHub.
* [x] Create first commit.
* [x] Push project to GitHub.

---

# PHASE 1 — Understand Our Data

* [x] Select initial character classes.
* [x] Start with `ሀ`.
* [x] Add `ለ`.
* [x] Add `መ`.
* [x] Create one folder per class.
* [x] Understand that folder names act as labels.
* [x] Locate Ethiopic-capable fonts.
* [x] Find Abyssinica SIL.
* [x] Find Nyala.
* [x] Generate a blank 64 × 64 image.
* [x] Draw an Ethiopic character using Pillow.
* [x] Save the character as PNG.
* [x] Measure character bounding box.
* [x] Calculate horizontal center.
* [x] Calculate vertical center.
* [x] Center characters automatically.
* [x] Make the character configurable.
* [x] Make the output folder configurable.
* [x] Generate `ሀ`.
* [x] Generate `ለ`.
* [x] Generate `መ`.

---

# PHASE 2 — Generate a Small Dataset

* [x] Create `generate_dataset.py`.
* [x] Generate characters using Abyssinica SIL.
* [x] Generate characters using Nyala.
* [x] Generate several font sizes.
* [x] Create approximately 10 synthetic images per class.
* [x] Verify approximately 33 total images.
* [ ] Inspect every generated image visually.
* [ ] Remove broken or clipped samples if found.
* [ ] Verify all classes contain the expected number of samples.

---

# PHASE 3 — Understand Pixels and Tensors

* [x] Open one generated image with Pillow.
* [x] Inspect image dimensions.
* [x] Inspect image mode.
* [x] Convert image to PyTorch tensor.
* [x] Inspect tensor values.
* [x] Understand grayscale values.
* [x] Understand `0.0` as black.
* [x] Understand `1.0` as white.
* [x] Understand tensor shape `[1, 64, 64]`.
* [x] Understand channel.
* [x] Understand image height.
* [x] Understand image width.
* [x] Understand that the neural network sees numbers, not letters.

---

# PHASE 4 — Understand Classes and Labels

* [x] Load the dataset using `ImageFolder`.
* [x] Detect class folders.
* [x] Convert character classes to numbers.
* [x] Confirm:

```text
ሀ → 0
ለ → 1
መ → 2
```

* [x] Understand the difference between image and label.
* [x] Understand that one training sample contains:

```text
tensor + correct label
```

* [x] Force ImageFolder images to grayscale.
* [x] Confirm tensor shape `[1, 64, 64]`.

---

# PHASE 5 — Split the Dataset

* [x] Understand why training and testing on the same data is misleading.
* [x] Create training indices.
* [x] Create validation indices.
* [x] Create test indices.
* [x] Use stratification.
* [x] Make the split reproducible with a random seed.
* [x] Produce approximately:

```text
23 training
5 validation
5 test
```

* [ ] Later redesign splitting when the dataset becomes larger.
* [ ] Later prevent synthetic variants from leaking across splits.
* [ ] Later perform writer-level splitting for handwriting.

---

# PHASE 6 — Understand Batches

* [x] Create a DataLoader.
* [x] Set batch size to 4.
* [x] Enable shuffling.
* [x] Load the first batch.
* [x] Inspect image batch shape.
* [x] Inspect label batch.
* [x] Understand:

```text
[4, 1, 64, 64]
```

as:

```text
4 images
1 channel
64 height
64 width
```

* [ ] Later experiment with different batch sizes.
* [ ] Understand how batch size affects memory.
* [ ] Understand how batch size affects gradient estimates.

---

# PHASE 7 — First Neural Network

* [x] Create `SimpleModel`.
* [x] Extend `nn.Module`.
* [x] Create a flatten layer.
* [x] Understand:

```text
64 × 64 = 4096
```

* [x] Create:

```text
4096 inputs
→
3 outputs
```

* [x] Print the model architecture.
* [x] Understand the forward method.
* [x] Understand a forward pass.
* [x] Pass fake image tensors through the model.
* [x] Inspect output shape `[4, 3]`.

---

# PHASE 8 — Understand Logits and Predictions

* [x] Understand that the model produces raw scores.
* [x] Learn the term **logit**.
* [x] Understand that logits are not probabilities.
* [x] Produce three logits per image.
* [x] Associate the three positions with three classes.
* [x] Use `argmax`.
* [x] Convert logits into predicted class IDs.
* [x] Convert class IDs back into Ethiopic characters.
* [x] Understand that the current model still has random weights.

---

# PHASE 9 — Understand Loss

* [x] Create example model outputs.
* [x] Create known correct labels.
* [x] Understand that the correct answer is one class ID.
* [x] Understand that we are not comparing three predefined scores to three model scores.
* [x] Create `CrossEntropyLoss`.
* [x] Calculate loss.
* [x] Obtain approximately:

```text
Loss = 0.3144
```

* [x] Understand that lower loss generally means a better prediction.
* [x] Understand that loss provides the learning signal for supervised learning.

---

# PHASE 10 — Understand Backpropagation

* [x] Enable gradient tracking.
* [x] Call:

```python
loss.backward()
```

* [x] Inspect gradients.
* [x] Understand that gradients describe how values affect loss.
* [x] Understand that gradients are directions, not weight updates.
* [x] Understand that backpropagation calculates gradients.
* [x] Understand that the model still has not been properly trained.

---

# PHASE 11 — Optimizer

CURRENT PHASE.

* [ ] Understand what an optimizer is.
* [ ] Understand why gradients alone do not modify weights.
* [ ] Introduce gradient descent.
* [ ] Introduce stochastic gradient descent.
* [ ] Create a simple optimizer.
* [ ] Start with SGD if useful for learning.
* [ ] Understand learning rate.
* [ ] Understand:

```python
optimizer.zero_grad()
```

* [ ] Understand why old gradients must be cleared.
* [ ] Calculate prediction.
* [ ] Calculate loss.
* [ ] Call `loss.backward()`.
* [ ] Inspect gradients before update.
* [ ] Call:

```python
optimizer.step()
```

* [ ] Inspect weights after update.
* [ ] Prove that model parameters actually changed.
* [ ] Calculate loss again.
* [ ] Observe whether one update reduces the loss.
* [ ] Experiment with learning rate.
* [ ] Later compare SGD and Adam.

---

# PHASE 12 — First Complete Training Step

* [ ] Load one real batch from our Ethiopic dataset.
* [ ] Send images through the model.
* [ ] Generate logits.
* [ ] Compare logits with labels.
* [ ] Calculate loss.
* [ ] Clear previous gradients.
* [ ] Backpropagate loss.
* [ ] Update model weights.
* [ ] Calculate predictions.
* [ ] Calculate batch accuracy.
* [ ] Print the result.

First complete learning operation:

```text
real Ethiopic images
↓
model
↓
logits
↓
loss
↓
backpropagation
↓
optimizer
↓
updated model
```

---

# PHASE 13 — Training Loop

* [ ] Understand an epoch.
* [ ] Understand iteration.
* [ ] Loop over all training batches.
* [ ] Clear gradients for each batch.
* [ ] Perform forward pass.
* [ ] Calculate loss.
* [ ] Perform backward pass.
* [ ] Update model.
* [ ] Track total loss.
* [ ] Track correct predictions.
* [ ] Calculate training accuracy.
* [ ] Finish one epoch.
* [ ] Repeat multiple epochs.
* [ ] Print results per epoch.

Expected output may resemble:

```text
Epoch 1
Loss: 1.07
Accuracy: 39%

Epoch 2
Loss: 0.91
Accuracy: 57%

Epoch 3
Loss: 0.70
Accuracy: 70%
```

Exact numbers will vary.

---

# PHASE 14 — Observe Learning

* [ ] Record training loss.
* [ ] Record training accuracy.
* [ ] Plot loss over epochs.
* [ ] Plot accuracy over epochs.
* [ ] Observe whether loss decreases.
* [ ] Observe whether accuracy increases.
* [ ] Investigate unstable training.
* [ ] Experiment with learning rate.
* [ ] Experiment with number of epochs.
* [ ] Understand convergence.

---

# PHASE 15 — Validation

* [ ] Put model into evaluation mode.
* [ ] Disable unnecessary gradient calculations.
* [ ] Run validation images through model.
* [ ] Calculate validation loss.
* [ ] Calculate validation accuracy.
* [ ] Compare training and validation metrics.
* [ ] Learn how validation detects generalization problems.

---

# PHASE 16 — Overfitting and Underfitting

* [ ] Understand overfitting.
* [ ] Understand underfitting.
* [ ] Identify training accuracy that is much higher than validation accuracy.
* [ ] Determine whether the model memorizes synthetic images.
* [ ] Expand dataset when necessary.
* [ ] Improve augmentation when necessary.
* [ ] Adjust model complexity.
* [ ] Explore regularization later.

---

# PHASE 17 — Final Test Evaluation

* [ ] Freeze development decisions.
* [ ] Load test dataset.
* [ ] Run model on test images.
* [ ] Calculate final test accuracy.
* [ ] Avoid tuning repeatedly on test results.
* [ ] Record test metrics.

---

# PHASE 18 — Error Analysis

* [ ] Identify incorrect predictions.
* [ ] Display incorrectly classified images.
* [ ] Show predicted label.
* [ ] Show correct label.
* [ ] Find commonly confused characters.
* [ ] Create confusion matrix.
* [ ] Calculate per-class accuracy.
* [ ] Investigate whether errors come from:

  * font
  * size
  * centering
  * clipping
  * character similarity
  * insufficient data

---

# PHASE 19 — Save the Model

* [ ] Understand model parameters.
* [ ] Understand weights.
* [ ] Understand biases.
* [ ] Understand `state_dict`.
* [ ] Save learned weights.
* [ ] Create `models/` directory.
* [ ] Save trained model checkpoint.
* [ ] Record model configuration.
* [ ] Record class mapping.
* [ ] Record training metrics.

---

# PHASE 20 — Load the Model

* [ ] Recreate model architecture.
* [ ] Load saved weights.
* [ ] Put model into evaluation mode.
* [ ] Confirm identical predictions.
* [ ] Understand difference between:

  * source code
  * model architecture
  * trained parameters
  * dataset

---

# PHASE 21 — Real Inference

Create something conceptually like:

```text
python predict.py image.png
```

Expected output:

```text
Prediction: ሀ
Confidence: 96.2%
```

Steps:

* [ ] Accept image path.
* [ ] Load image.
* [ ] Convert to grayscale.
* [ ] Resize appropriately.
* [ ] Convert to tensor.
* [ ] Add batch dimension.
* [ ] Run model.
* [ ] Obtain logits.
* [ ] Convert logits to probabilities.
* [ ] Select highest probability.
* [ ] Convert class number to character.
* [ ] Display prediction.

---

# PHASE 22 — Understand Probabilities

* [ ] Learn Softmax.
* [ ] Convert logits into probabilities.
* [ ] Understand why Softmax is unnecessary before `CrossEntropyLoss`.
* [ ] Display class probabilities.
* [ ] Understand confidence carefully.
* [ ] Learn that confidence does not guarantee correctness.

---

# PHASE 23 — Improve Dataset Quality

* [ ] Add more Ethiopic fonts.
* [ ] Add more font sizes.
* [ ] Add controlled horizontal movement.
* [ ] Add controlled vertical movement.
* [ ] Add slight rotations.
* [ ] Add contrast variation.
* [ ] Add brightness variation.
* [ ] Add mild blur.
* [ ] Add mild noise.
* [ ] Add background variation.
* [ ] Add printing artifacts.
* [ ] Add scanning artifacts.
* [ ] Avoid unrealistic transformations.
* [ ] Avoid data leakage.

---

# PHASE 24 — Expand Character Classes

Progress gradually:

```text
3 classes
↓
5 classes
↓
10 classes
↓
30 classes
↓
larger Ethiopic set
```

For every expansion:

* [ ] Check dataset balance.
* [ ] Generate sufficient examples.
* [ ] Retrain model.
* [ ] Measure accuracy.
* [ ] Examine confusion matrix.
* [ ] Identify visually similar characters.

---

# PHASE 25 — Build a Convolutional Neural Network

* [ ] Understand convolution.
* [ ] Understand kernel/filter.
* [ ] Understand local receptive fields.
* [ ] Understand feature maps.
* [ ] Understand stride.
* [ ] Understand padding.
* [ ] Understand ReLU.
* [ ] Understand pooling.
* [ ] Build first convolutional layer.
* [ ] Inspect convolution output shape.
* [ ] Add activation.
* [ ] Add pooling.
* [ ] Add second convolution.
* [ ] Flatten learned features.
* [ ] Add fully connected classifier.
* [ ] Train CNN.
* [ ] Compare CNN against linear baseline.

---

# PHASE 26 — Visualize What the CNN Learns

* [ ] Visualize filters.
* [ ] Visualize feature maps.
* [ ] Observe edge detection.
* [ ] Observe stroke detection.
* [ ] Investigate deeper features.
* [ ] Understand hierarchical feature learning.

---

# PHASE 27 — Proper Experiment Tracking

* [ ] Store training configuration.
* [ ] Record learning rate.
* [ ] Record batch size.
* [ ] Record epochs.
* [ ] Record dataset version.
* [ ] Record model architecture.
* [ ] Record random seed.
* [ ] Save metrics.
* [ ] Save graphs.
* [ ] Compare experiments.
* [ ] Keep experiments reproducible.

---

# PHASE 28 — Handwritten Character Dataset (future; not in this stage)

* [ ] Design data collection format.
* [ ] Collect handwriting from multiple people.
* [ ] Obtain appropriate consent for collected data.
* [ ] Assign anonymous writer IDs.
* [ ] Scan or photograph handwriting.
* [ ] Segment individual characters.
* [ ] Normalize images.
* [ ] Verify labels manually.
* [ ] Split by writer rather than random image.
* [ ] Train printed-only model.
* [ ] Train handwritten model.
* [ ] Compare results.

---

# PHASE 29 — Printed Character Segmentation (first version complete)

The classifier itself assumes:

```text
one image = one character
```

The new OpenCV layer now finds characters inside larger printed images.

* [x] Convert images to grayscale.
* [x] Detect foreground polarity.
* [x] Apply Otsu thresholding and conditional adaptive thresholding.
* [x] Remove small noise with morphology.
* [x] Detect connected components.
* [x] Group plausible disconnected pieces by overlap, distance, and size.
* [x] Split unusually wide components using vertical projection valleys.
* [x] Identify character boundaries and reading order.
* [x] Reuse the shared CNN preprocessing function for every crop.
* [x] Feed batches of segmented crops into the active classifier.
* [x] Handle different clean printed fonts, sizes, whitespace, and shifts in tests.
* [x] Draw inspectable word and character bounding boxes.
* [ ] Improve touching-character separation using real observed failures.
* [ ] Add skew and perspective correction for camera images.

---

# PHASE 30 — Printed Word Recognition (first version complete)

* [x] Process multiple detected characters.
* [x] Run the active 290-class CNN on every crop.
* [x] Reconstruct character sequences.
* [x] Preserve left-to-right Ethiopic reading order.
* [x] Recognize clean generated printed words.
* [x] Store bounding box, prediction, confidence, alternatives, and crop.
* [x] Flag low-confidence characters without discarding the raw prediction.
* [x] Expose segmentation errors separately from classifier errors.
* [x] Verify `ሰላም` end to end with the active checkpoint.
* [ ] Create a larger independent real-word benchmark and report word accuracy.
* [ ] Improve recovery for touching or broken printed characters.

---

# PHASE 31 — Printed Line Recognition (first version complete)

Instead of:

```text
single image → single character
```

The current first version uses:

```text
printed line image
↓
relative line, gap, word, and character segmentation
↓
active character CNN
↓
reconstructed text sequence
```

* [x] Group characters into lines using relative vertical overlap.
* [x] Infer word boundaries from relative component width and clustered gaps.
* [x] Preserve top-to-bottom line and left-to-right character order.
* [x] Restore spaces and line breaks.
* [x] Display per-character crops, confidence, and uncertainty.
* [x] Keep optional translation separate from OCR.
* [x] Test a generated two-word printed line end to end.
* [ ] Benchmark larger real printed lines and punctuation.
* [ ] Add full-page region and layout analysis.

Possible later sequence architectures:

* CNN + recurrent network
* CTC-based recognition
* transformer-based recognition
* vision transformer approaches

These remain future experiments; the current implementation intentionally
reuses the trained character CNN.

---

# PHASE 32 — Amharic Language Model Integration (future; not translation)

Vision sometimes makes mistakes between visually similar characters.

A language model can provide context.

Example:

```text
vision prediction
+
surrounding characters
+
Amharic word probability
↓
corrected text
```

Future tasks:

* [ ] Build Amharic vocabulary.
* [ ] Collect legal/open Amharic corpora.
* [ ] Train character-level language model.
* [ ] Train word-level language model.
* [ ] Combine OCR confidence with linguistic probability.
* [ ] Correct likely OCR mistakes.

---

# PHASE 33 — Full-Page Amharic OCR (future)

Target system:

```text
photo / scan / document
↓
image preprocessing
↓
text detection
↓
line segmentation
↓
character / sequence recognition
↓
language correction
↓
Amharic text
```

Possible inputs:

* scanned documents
* books
* handwritten notes
* signs
* forms
* historical documents
* photographs

---

# PHASE 34 — Production API (future)

* [ ] Build inference service.
* [ ] Create API endpoint.
* [ ] Accept uploaded image.
* [ ] Run preprocessing.
* [ ] Run model.
* [ ] Return prediction.
* [ ] Return confidence.
* [ ] Add error handling.
* [ ] Add logging.
* [ ] Add tests.
* [ ] Benchmark inference time.

Possible stack:

```text
Python
PyTorch
FastAPI
```

---

# PHASE 35 — Desktop User Interface (complete for current OCR stage)

Implemented application:

```text
Character, word, or line image
↓
Original/binary/segmentation preview
↓
Local CNN recognition
↓
Traceable Ethiopic result and confidence
↓
Optional English translation
```

* [x] Maximized professional light desktop layout.
* [x] Character dataset browser and external upload.
* [x] Correct answer, confidence, top predictions, and correct/wrong state.
* [x] Word OCR and Sentence OCR navigation.
* [x] Original, binary, and bounding-box previews.
* [x] Low-confidence selector and raw per-crop diagnostics.
* [x] Held-out evaluation page.
* [x] Checkpoint-derived model information page.
* [x] Educational pipeline visualization page.
* [x] Real accuracy, loss, and learning-rate graph page.
* [x] Optional translation and copy controls with failure isolation.
* [ ] Consider web, mobile, or document-processing interfaces later.

---

# PHASE 36 — Advanced Ethiopian-Language AI Projects

Skills from this project could later support:

* [ ] Amharic document digitization
* [ ] handwritten Amharic transcription
* [ ] historical manuscript processing
* [ ] Ethiopian archive search
* [ ] OCR-assisted translation
* [ ] Amharic document question answering
* [ ] intelligent form processing
* [ ] receipt and invoice extraction
* [ ] Amharic educational applications
* [ ] speech-to-text
* [ ] text-to-speech
* [ ] multilingual Ethiopian-language NLP
* [ ] Amharic scam/SMS classification
* [ ] document classification
* [ ] handwriting assistance
* [ ] accessibility systems
* [ ] multimodal Ethiopian-language AI

---

# PHASE 37 — Optional Translation Integration (complete for current stage)

Translation is downstream from OCR. It does not replace or assist visual
recognition.

* [x] Define a provider interface separate from OCR logic.
* [x] Use a documented keyless Amharic-to-English endpoint for the normal demo.
* [x] Send reconstructed text only after an explicit Translate action.
* [x] Keep images, crops, logits, and model data local.
* [x] Use timeouts and UTF-8 byte-aware request chunking.
* [x] Preserve the Amharic OCR result when translation fails.
* [x] Keep endpoint and optional contact email configurable through environment variables.
* [ ] Re-evaluate provider limits and terms before public/high-volume deployment.

---


# Ultimate Goal

The final objective is not simply:

```text
"I built an Amharic OCR application."
```

The objective is to be able to explain:

```text
how images become tensors
how datasets become supervised examples
how neural networks produce logits
how loss measures mistakes
how gradients are calculated
how optimizers change weights
how training improves the model
how validation measures generalization
how CNNs learn visual features
how OCR grows from character recognition
how language models can improve visual recognition
```

The project should grow only as fast as these concepts are genuinely understood.

## Completed milestone: Full Printed Character Expansion
**Objective**: Expand the CNN to recognize the full canonical Amharic printed character inventory without losing previously learned capabilities.

**Changes made**:
- Expanded dataset generation to output 290 classes, explicitly listed in `src/characters.json`.
- Targeted 1200 high-quality variants per class.
- Re-architected GUI into a tabbed layout, hiding dense details under informative tabs ("Predict", "Test Model", "Model Information").
- Safe dynamic CNN initialization (`num_classes=290`) rejecting old incompatible weights for fresh retraining.
- Organized older linear models into `src/legacy_linear/` to preserve learning history without cluttering the active pipeline.
