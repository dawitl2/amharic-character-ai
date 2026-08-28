
## Phase 28: Full Printed Character Expansion
**Objective**: Expand the CNN to recognize the full canonical Amharic printed character inventory without losing previously learned capabilities.

**Changes made**:
- Expanded dataset generation to output 290 classes, explicitly listed in `src/characters.json`.
- Targeted 1200 high-quality variants per class.
- Re-architected GUI into a tabbed layout, hiding dense details under informative tabs ("Predict", "Test Model", "Model Information").
- Safe dynamic CNN initialization (`num_classes=290`) rejecting old incompatible weights for fresh retraining.
- Organized older linear models into `src/legacy_linear/` to preserve learning history without cluttering the active pipeline.
