import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    ce = nn.CrossEntropyLoss()
    for waves, labels in tqdm(dataloader, desc="Train batch"):
        waves = waves.to(device)
        labels = labels.to(device)
        logits = model(waves)
        loss = ce(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)
