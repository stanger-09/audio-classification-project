"""
Evaluate model on test dataset
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(str(Path(__file__).parent.parent))

from config import *
from utils.dataset import AudioDataset
from models.wav2vec_classifier import Wav2VecClassifier
from training.collate_fn import collate_fn
from training.accuracy import calculate_accuracy, calculate_per_class_accuracy


def load_trained_model():
    """Load model from checkpoint"""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    num_classes = checkpoint['num_classes']
    model = Wav2VecClassifier(num_classes=num_classes).to(DEVICE)
    
    # Load weights
    model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    model.attention.load_state_dict(checkpoint['attention_state_dict'])
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    model.eval()
    print(f"✅ Model loaded from {CHECKPOINT_PATH}")
    
    return model


def main():
    print("\n" + "="*60)
    print("📊 ASiT Model Evaluation on Test Set")
    print("="*60 + "\n")
    
    # Load dataset
    print("📂 Loading dataset...")
    dataset = AudioDataset(DATASET_ROOT)
    
    # Split (same as training)
    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val = int(VAL_SPLIT * n_total)
    n_test = n_total - n_train - n_val
    
    _, _, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)  # Same seed as training!
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    print(f"Test set: {n_test} samples\n")
    
    # Load model
    model = load_trained_model()
    
    # Evaluate
    print("🔍 Evaluating on test set...")
    test_acc = calculate_accuracy(model, test_loader)
    print(f"\n✅ Overall Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    
    # Per-class accuracy
    print("="*60)
    print("📈 Per-Class Accuracy:")
    print("="*60)
    
    per_class_acc = calculate_per_class_accuracy(
        model, test_loader, len(dataset.class_to_idx)
    )
    
    for idx, class_name in enumerate(dataset.class_names):
        acc = per_class_acc.get(idx, 0.0)
        bar = "█" * int(acc * 20)
        print(f"{class_name:20s}: {acc*100:5.2f}% {bar}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()