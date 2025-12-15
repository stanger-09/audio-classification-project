"""
Main training script - Entry point for model training
"""
import torch
from torch.utils.data import DataLoader, random_split
import sys
from pathlib import Path

# 🔥 NEW imports for visualization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from utils.dataset import AudioDataset
from models.wav2vec_classifier import Wav2VecClassifier
from training.train_full import train_model
from training.collate_fn import collate_fn
from training.accuracy import calculate_accuracy, calculate_per_class_accuracy


def main():
    """Main training pipeline"""

    print("\n" + "="*60)
    print("🎵 ASiT Audio Classification Training")
    print("="*60 + "\n")

    # Load dataset
    print("📂 Loading dataset...")
    dataset = AudioDataset(DATASET_ROOT)

    # Verify 21 classes
    if len(dataset.class_to_idx) != 21:
        print(f"⚠️  WARNING: Expected 21 classes but found {len(dataset.class_to_idx)}")

    # Split dataset
    n_total = len(dataset)
    n_train = int(TRAIN_SPLIT * n_total)
    n_val = int(VAL_SPLIT * n_total)
    n_test = n_total - n_train - n_val

    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {n_train:5d} samples ({TRAIN_SPLIT*100:.0f}%)")
    print(f"   Validation: {n_val:5d} samples ({VAL_SPLIT*100:.0f}%)")
    print(f"   Test:       {n_test:5d} samples ({TEST_SPLIT*100:.0f}%)\n")

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Initialize model
    print("🤖 Initializing model...")
    model = Wav2VecClassifier(num_classes=len(dataset.class_to_idx)).to(DEVICE)
    print(f"   Model: Wav2Vec2 with {len(dataset.class_to_idx)} classes")
    print(f"   Device: {DEVICE}")

    # Freeze or unfreeze backbone
    if not FINETUNE_BACKBONE:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    # Setup optimizer
    params = [
        {"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER},
        {"params": model.attention.parameters(), "lr": LEARNING_RATE_CLASSIFIER},
    ]

    if FINETUNE_BACKBONE:
        params.append({"params": model.backbone.parameters(), "lr": LEARNING_RATE_BACKBONE})

    optimizer = torch.optim.Adam(params)

    print(f"   Optimizer: Adam")
    print(f"   Classifier LR: {LEARNING_RATE_CLASSIFIER}")
    if FINETUNE_BACKBONE:
        print(f"   Backbone LR: {LEARNING_RATE_BACKBONE}")

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=EPOCHS
    )

    # Final test evaluation
    print("📊 Evaluating on test set...")
    test_acc = calculate_accuracy(model, test_loader)
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")

    # Per-class accuracy
    print("📈 Per-Class Accuracy:")
    per_class_acc = calculate_per_class_accuracy(
        model, test_loader, len(dataset.class_to_idx)
    )

    for idx, class_name in enumerate(dataset.class_names):
        acc = per_class_acc.get(idx, 0.0)
        print(f"   {class_name:20s}: {acc*100:5.2f}%")

    print("\n✅ Training pipeline completed successfully!\n")

    # 🔥 RETURN objects for demo visualization
    return model, dataset, test_loader, history


# ================== VISUALIZATION ==================

def plot_confusion_matrix(model, loader, class_names):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds = torch.argmax(model(x), dim=1).cpu()
            y_true.extend(y)
            y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        cmap="Blues", xticks_rotation=90
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.show()


def plot_learning_curves(history):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], marker="o")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


# ================== ENTRY ==================

if __name__ == "__main__":
    model, dataset, test_loader, history = main()

    # 🔥 DEMO OUTPUTS
    plot_confusion_matrix(model, test_loader, dataset.class_names)
    plot_learning_curves(history)
