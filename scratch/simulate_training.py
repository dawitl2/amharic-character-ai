import sys, os
from pathlib import Path

# Add src to path
sys.path.append(str(Path('src').resolve()))
from training import TrainingSettings, run_training

sys.stdout.reconfigure(encoding='utf-8')

settings = TrainingSettings(
    max_epochs=1,
    batch_size=32,
    learning_rate=0.01,
    scheduler_patience=10,
    early_stopping_patience=10,
)
try:
    run_training(settings)
except Exception as e:
    import traceback
    traceback.print_exc()
