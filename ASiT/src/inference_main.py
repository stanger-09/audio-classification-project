import torch
from pathlib import Path
from config import DEVICE, WAV2VEC_MODEL, DATASET_ROOT
from models.wav2vec_classifier import W2VClassifier
from inference.predict import predict

CHECKPOINT_PATH = "best_wav2vec_classifier.pt"

def main():
    dataset_root = Path(DATASET_ROOT)
    class_names = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    num_classes = len(class_names)

    model = W2VClassifier(WAV2VEC_MODEL, num_classes).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model.backbone.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Available classes:", class_names)

    test_wav = "/root/dataset/augmented-audio/Les Brown/0Les Brown0.wav"
    predict(model, test_wav, class_names, DEVICE)

if __name__ == "__main__":
    main()
