import sys, os
from pathlib import Path

# Add src to path
sys.path.append(str(Path('src').resolve()))
from inference import InferenceEngine
from cnn_model import CharacterCNN
import torch

sys.stdout.reconfigure(encoding='utf-8')

try:
    engine = InferenceEngine.from_artifacts()
    print(f"Loaded InferenceEngine. Classes: {len(engine.bundle.idx_to_class)}")
    print(f"CNN Classes configured: {engine.bundle.model.num_classes}")
    
    # Test forward pass
    fake_img = torch.randn(1, 1, 64, 64)
    logits = engine.bundle.model(fake_img)
    print(f"Logits shape: {logits.shape}")
    
    sample_file = list(Path('data').glob('*/*.png'))[0]
    pred = engine.predict_path(sample_file)
    print(f"Predicted {sample_file.name}: {pred.predicted_character} with {pred.confidence:.2f} conf")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAILED:", e)
