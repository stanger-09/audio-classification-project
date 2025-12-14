"""
Central configuration file for the entire project
"""
import torch
from pathlib import Path

# ============ PATHS ============
PROJECT_ROOT = Path(__file__).parent
DATASET_ROOT = r"C:\Users\ADMIN\OneDrive\Desktop\Project_3.1\W2v2\DATASET"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============ AUDIO CONFIG ============
RESAMPLE_RATE = 16000
AUDIO_DURATION = 5  # seconds
TOTAL_SAMPLES = RESAMPLE_RATE * AUDIO_DURATION

# ============ MODEL CONFIG ============
WAV2VEC_MODEL = "facebook/wav2vec2-base"
CHECKPOINT_PATH = CHECKPOINT_DIR / r"C:\Users\ADMIN\OneDrive\Desktop\ASiT\ASiT\best_wav2vec_classifier.pt"

# ============ TRAINING CONFIG ============
BATCH_SIZE = 8
NUM_WORKERS = 2
LEARNING_RATE_CLASSIFIER = 1e-3
LEARNING_RATE_BACKBONE = 1e-5
EPOCHS = 2
FINETUNE_BACKBONE = False  # Set True when ready

# ============ DATA SPLIT ============
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============ DEVICE ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Configuration Loaded")
print(f"   Device: {DEVICE}")
print(f"   Dataset: {DATASET_ROOT}")
print(f"   Checkpoint: {CHECKPOINT_PATH}")