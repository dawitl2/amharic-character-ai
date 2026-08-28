# Ethiopic OCR — Current Project Handout

## What have we built?

We have built an educational supervised-learning system for printed Amharic/Ethiopic OCR.

It now supports:

```text
single character image
→ our trained CNN
→ one of 290 Ethiopic characters

printed word image
→ OpenCV character segmentation
→ our trained CNN for every crop
→ reconstructed word

printed sentence/line image
→ OpenCV word and character segmentation
→ our trained CNN for every crop
→ reconstructed Amharic text
→ optional external English translation
```

The OCR intelligence is ours: the local `CharacterCNN` was trained from the project's labeled dataset. The optional translation provider receives reconstructed text only. It does not inspect images and does not perform OCR.

## Why are we building it?

OCR products already exist. This project is designed to learn how intelligence is trained rather than using an OCR model as a black box.

The central idea is **supervised learning**:

```text
image tensor + correct label
↓
CNN logits
↓
cross-entropy loss
↓
backpropagation and gradients
↓
optimizer updates weights
↓
the model improves through repetition
```

## Current active model

| Item | Saved value |
| --- | ---: |
| Architecture | CharacterCNN |
| Classes | 290 |
| Parameters | 1,090,914 |
| Input | 64 × 64 grayscale |
| Dataset | 348,000 images |
| Train / validation / test | 243,600 / 52,200 / 52,200 |
| Completed epochs | 27 |
| Epoch 27 training accuracy | 97.4224% |
| Best validation accuracy | 93.9751% |
| Best epoch | 27 |
| Full independent test accuracy | Not yet recorded |

The previous training window was closed before the run reached its final-test step. The best and latest CNN checkpoints remain intact, so training can resume at epoch 28.

## Character recognition

The CNN receives a tensor shaped like:

```text
[batch, 1, 64, 64]
```

Its architecture is:

```text
64×64 grayscale
→ Conv 1→16
→ ReLU
→ MaxPool
→ Conv 16→32
→ ReLU
→ MaxPool
→ Flatten 8,192
→ Linear 128
→ ReLU
→ 290 logits
→ character
```

The logits are raw class scores. Softmax turns them into inference probabilities. A high probability is confidence, not proof that the answer is correct.

## Word and sentence OCR

OpenCV and the CNN have different jobs:

```text
OpenCV = WHERE is each word or character?
CNN    = WHAT character is inside each crop?
```

Segmentation uses:

- grayscale and polarity detection;
- Otsu thresholding and adaptive thresholding when lighting is uneven;
- morphological noise removal;
- connected components;
- grouping of plausible disconnected glyph pieces;
- vertical projection valleys for unusually wide components;
- relative line, spacing, and word-gap analysis;
- top-to-bottom, left-to-right reading order.

It does not assume one contour always equals one character.

## Shared preprocessing

All external characters and segmented crops reuse `src/preprocessing.py`:

```text
grayscale
→ remove surrounding whitespace
→ preserve aspect ratio
→ fit glyph to the training-like scale
→ center on a square 64×64 canvas
→ convert to tensor
→ apply the checkpoint's normalization
```

Keeping one preprocessing contract prevents the GUI, tests, and OCR engine from sending different kinds of tensors to the model.

## Diagnosing a wrong word

The GUI deliberately shows blue word boxes, green character boxes, each crop, the raw prediction, confidence, and uncertainty state.

```text
wrong number/order of boxes
→ segmentation problem

correct boxes but wrong character
→ CNN/generalization problem

correct raw character but displayed ?
→ confidence threshold decision
```

Low-confidence characters are not silently presented as reliable. The readable result can show `?`, while the raw CNN guess remains visible.

## Translation

Translation is optional and modular:

```text
our OCR engine
→ reconstructed Amharic text
→ MyMemory translation provider
→ English text
```

Only text is sent when **Translate** is selected. If the internet or provider fails, the Amharic OCR result remains available.

## Desktop application

The light, maximized desktop interface has these sections:

- **Character** — dataset browser, upload, confidence, top predictions, correctness;
- **Word OCR** — upload, segmentation preview, word, crops, confidence, translation;
- **Sentence OCR** — word/character boxes, reading order, text, translation;
- **Evaluate** — random labeled samples from the selected split;
- **Model Info** — checkpoint-derived facts and metrics;
- **Pipeline** — visual explanations of character, word, and sentence processing;
- **Training Graphs** — real saved accuracy, loss, and learning-rate history.

## Where are we now?

```text
labeled synthetic data                 complete
linear learning baseline               preserved
290-class CNN                           trained through epoch 27
single-character application           complete
printed word segmentation              first robust version complete
printed sentence/line segmentation     first robust version complete
traceable CNN reconstruction           complete
optional text translation              complete
handwriting                             future work
full-page arbitrary document OCR       future work
```

## Run the application

From Windows Command Prompt:

```bat
cd /d "C:\Users\enkud\Desktop\Projects\AI\amharic-character-ai"
.venv\Scripts\python.exe src\gui.py
```

## Resume the interrupted training run

This resumes from the compatible latest checkpoint. It does not erase epoch 27:

```bat
cd /d "C:\Users\enkud\Desktop\Projects\AI\amharic-character-ai"
.venv\Scripts\python.exe src\train.py --max-epochs 200
```

`--fresh` should be used only when intentionally starting again from random weights. Fresh mode first archives the active artifacts for recovery.

## What happens next?

The next immediate step is to resume training and let the run complete its independent test evaluation. After that, test printed words and lines from real scanners/cameras and expand segmentation tests around observed failures.

Handwriting remains a later stage. When introduced, printed and handwritten examples should be mixed so improved handwriting support does not replace the working printed capability.
