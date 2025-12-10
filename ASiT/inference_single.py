import sys
import os

# Make src/ importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
from pathlib import Path

from src.config import DEVICE, WAV2VEC_MODEL, DATASET_ROOT
from src.models.wav2vec_classifier import W2VClassifier
from src.inference.predict import predict


# USE RAW STRING for checkpoint path
CHECKPOINT = r"C:\Users\SAKETH BUDARAPU\Desktop\ASiT\best_wav2vec_classifier.pt"
# or: CHECKPOINT = r"best_wav2vec_classifier.pt" if it's in ASiT folder


def main(wav_path):
    # load class names
    dataset_root = Path(DATASET_ROOT)
    class_names = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    num_classes = len(class_names)

    # load model
    model = W2VClassifier(WAV2VEC_MODEL, num_classes).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.backbone.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])
    model.eval()

    # run prediction
    predict(model, wav_path, class_names, DEVICE)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference_single.py path/to/file.wav")
    else:
        main(sys.argv[1])
