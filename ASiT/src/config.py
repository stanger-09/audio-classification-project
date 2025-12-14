# ---------------- CONFIG ----------------
DATASET_ROOT = r""
RESAMPLE = 16000
SECONDS = 5
TOTAL_SAMPLES = RESAMPLE * SECONDS

BATCH_SIZE = 8
NUM_WORKERS = 2

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WAV2VEC_MODEL = "facebook/wav2vec2-base"
LR_CLASSIFIER = 1e-3
LR_BACKBONE = 1e-5
EPOCHS_CLASSIFIER = 10
FINETUNE_BACKBONE = False
