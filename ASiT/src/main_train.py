import os
import torch
from torch.utils.data import DataLoader, random_split

from config import *
from dataset.raw_audio_dataset import RawAudioDataset, collate_fn
from models.wav2vec_classifier import W2VClassifier
from training.train_full import train_classifier_full

def main():
    ds = RawAudioDataset(DATASET_ROOT)
    if len(ds) == 0:
        raise RuntimeError("No audio files found. Fix DATASET_ROOT.")

    val_frac = 0.15
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=NUM_WORKERS)

    num_classes = len(set(ds.labels))
    model = W2VClassifier(wav2vec_model_name=WAV2VEC_MODEL, num_classes=num_classes).to(DEVICE)

    print("Model backbone hidden dim:", model.backbone.config.hidden_size)
    print("Num classes:", num_classes)
    print("Device:", DEVICE)

    train_classifier_full(
        model,
        train_loader,
        val_loader,
        DEVICE,
        epochs=EPOCHS_CLASSIFIER,
        lr_classifier=LR_CLASSIFIER,
        lr_backbone=LR_BACKBONE,
        finetune_backbone=FINETUNE_BACKBONE
    )

if __name__ == "__main__":
    main()
