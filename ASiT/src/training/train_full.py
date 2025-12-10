import torch
from training.train_epoch import train_epoch
from training.accuracy import compute_accuracy

def train_classifier_full(model, train_loader, val_loader, device,
                          epochs, lr_classifier, lr_backbone, finetune_backbone):

    if not finetune_backbone:
        model.freeze_backbone()
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr_classifier)
    else:
        model.unfreeze_backbone()
        optimizer = torch.optim.Adam([
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_classifier},
        ])

    best_val = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_acc = compute_accuracy(model, train_loader, device)
        val_acc = compute_accuracy(model, val_loader, device) if val_loader is not None else 0.0

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "backbone": model.backbone.state_dict(),
                "classifier": model.classifier.state_dict()
            }, "best_wav2vec_classifier.pt")

    print("Best Val Acc:", best_val)
