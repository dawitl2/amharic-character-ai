# Amharic / Ethiopic Character Recognition

A learning-oriented PyTorch project for classifying the 290 Ethiopic characters in `src/characters.json` with one active convolutional neural network and a small desktop research interface.

## Active pipeline

```text
image
→ shared grayscale preparation
→ [1, 1, 64, 64] tensor
→ CharacterCNN
→ raw class logits
→ softmax(logits, dim=1) for inference only
→ character decoded with the checkpoint class_to_idx mapping
```

The GUI, command-line prediction, automatic evaluation, and sanity diagnostics all load:

```text
models/cnn_model_config.json
models/best_cnn_model.pth
```

Loading is strict. Architecture, tensor names, input format, output count, preprocessing, and class mapping must agree. The application never falls back to random weights or the legacy linear model.

## CNN architecture

`CharacterCNN` contains two convolution/ReLU/max-pooling blocks followed by a 128-unit fully connected layer and one raw logit per class. The active input contract is 64 × 64 grayscale with values in `[0, 1]`; current normalization is the identity transform (mean `0`, standard deviation `1`).

## Training

Run from the repository root:

```powershell
.venv\Scripts\python.exe src\train.py
```

When the dataset or character mapping has been replaced, start a clean run with:

```powershell
.venv\Scripts\python.exe src\train.py --fresh
```

`--fresh` validates the dataset first, then moves old CNN checkpoints, metrics, configuration, and split data into a timestamped `models/archive/` directory. It does not delete them. Without `--fresh`, training resumes only from a strictly compatible latest checkpoint.

For CPU training, two persistent background image-loading workers can reduce the PNG decoding bottleneck:

```powershell
.venv\Scripts\python.exe src\train.py --max-epochs 200 --num-workers 2
```

Do not repeat `--fresh` after the new split manifest has already been created unless you intentionally want to archive it and rebuild the split. Split creation and long epochs display progress while they run.

Documented defaults:

- maximum cumulative epochs: 200
- optimizer: SGD
- learning rate: 0.01
- batch size: 64
- data-loader workers: 0 by default; `--num-workers 2` is the conservative CPU option
- loss: `CrossEntropyLoss` on raw logits
- scheduler: `ReduceLROnPlateau`, factor 0.5, patience 5, minimum LR 0.00001
- early stopping: patience 15, minimum validation-accuracy change 0.05 percentage points
- deterministic seed: 42
- split: 70% training, 15% validation, 15% test

`--max-epochs` is a cumulative ceiling, not a request for that many additional epochs. Training can finish earlier through early stopping. Run `src/train.py --help` for explicit overrides.

Training saves:

```text
models/best_cnn_model.pth        best validation checkpoint
models/latest_cnn_checkpoint.pth latest resumable state
models/cnn_model_config.json     GUI-readable model metadata
models/cnn_training_metrics.csv  per-epoch metrics
models/cnn_data_split.json       deterministic split membership
```

The latest checkpoint includes model, optimizer, scheduler, cumulative epoch, best score, and early-stopping state.

## Data split and validation limits

The split manifest is deterministic and stratified. Exact normalized duplicates and files with explicit source/augmentation naming are assigned as groups, so a known family cannot cross splits.

The current dataset was generated with a small shared set of fonts and procedural transformations. Its filenames do not preserve the generating font or an original-sample identifier. Therefore the measured validation/test accuracy is valid for held-out samples from the same synthetic generator distribution, but it is not evidence of equal performance on handwriting, camera images, scans, or unseen fonts. Future generators should save provenance such as `source_id`, font, writer, and augmentation parent so entire sources can be held out.

The bundled checkpoint was migrated from the historical per-image random split and predates `cnn_data_split.json`. Until a fresh training run uses the manifest, manifest-partition results are diagnostic rather than genuinely held-out from that checkpoint. The diagnostics and GUI state this explicitly instead of presenting those figures as independent validation/test accuracy.

## Diagnostics

Before a long training run, perform the read-only full dataset preflight:

```powershell
.venv\Scripts\python.exe src\preflight_training.py --all-images
```

This decodes every image, rejects blank images and identical content with conflicting labels, validates the canonical 290-class mapping, and checks the CNN output shape. It does not train or write model artifacts.

Run all three labeled splits separately:

```powershell
.venv\Scripts\python.exe sanity_test.py
```

Sample a smaller number from each split:

```powershell
.venv\Scripts\python.exe sanity_test.py --limit 100
```

Evaluate labeled external images arranged as `external/<character>/<image>`:

```powershell
.venv\Scripts\python.exe sanity_test.py --external-dir path\to\external
```

Training, validation, test, and external results are never combined. External accuracy is reported as N/A unless labeled examples are supplied.

## Prediction and GUI

```powershell
.venv\Scripts\python.exe src\predict.py path\to\image.png
.venv\Scripts\python.exe src\gui.py
```

Dataset images use the same shared resize/tensor path used by training. External images additionally receive whitespace removal and an aspect-preserving centered fit before that same tensor conversion. The GUI only shows CORRECT/WRONG when the image comes from the labeled dataset.

## Legacy linear experiments

The `LinearModel` and early phase scripts remain in the repository as learning history. They are not imported by the active CNN GUI, predictor, trainer, or diagnostics. Historical ambiguous artifacts such as `best_model_weights.pth`, `model_config.json`, and `simple_model_weights.pth` are not active model files.

To migrate a known CNN state dict from the historical ambiguous names once, run:

```powershell
.venv\Scripts\python.exe src\migrate_cnn_checkpoint.py
```

The migration validates convolutional tensor names and output size before writing the explicit active CNN artifacts.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -v
.venv\Scripts\python.exe test_gui_script.py
```

The GUI smoke test confirms that one image passed directly through the inference engine and through the GUI produces identical logits, probabilities, prediction, and confidence.
