"""
Single epoch training logic
"""
import torch
import torch.nn as nn
from config import DEVICE


def train_one_epoch(model, dataloader, optimizer, loss_fn):
    """
    Train model for one epoch
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
    
    Returns:
        average_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for audio, labels in dataloader:
        audio = audio.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        logits = model(audio)
        loss = loss_fn(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    average_loss = total_loss / max(num_batches, 1)
    return average_loss