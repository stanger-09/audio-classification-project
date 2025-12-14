"""
Complete training loop with validation and checkpointing
"""
import torch
import torch.nn as nn
from pathlib import Path
from .train_epoch import train_one_epoch
from .accuracy import calculate_accuracy
from config import DEVICE, CHECKPOINT_PATH


def train_model(model, train_loader, val_loader, optimizer, epochs):
    """
    Full training loop with validation and checkpointing
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer
        epochs: Number of epochs
    
    Returns:
        Dictionary with training history
    """
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'val_acc': []
    }
    
    print("\n" + "="*60)
    print("🚀 Starting Training")
    print("="*60 + "\n")
    
    for epoch in range(epochs):
        # Train for one epoch
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        
        # Validate
        val_acc = calculate_accuracy(model, val_loader)
        
        # Store history
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss:    {avg_loss:.4f}")
        print(f"  Val Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, val_acc)
            print(f"  ✅ New best model saved! (Val Acc: {val_acc*100:.2f}%)")
        
        print()
    
    print("="*60)
    print("✨ Training Complete!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("="*60 + "\n")
    
    return history


def save_checkpoint(model, val_accuracy):
    """Save model checkpoint"""
    # Get class mapping from model (stored during initialization)
    checkpoint = {
        'backbone_state_dict': model.backbone.state_dict(),
        'attention_state_dict': model.attention.state_dict(),
        'classifier_state_dict': model.classifier.state_dict(),
        'num_classes': model.num_classes,
        'val_accuracy': val_accuracy
    }
    
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"   💾 Checkpoint saved: {CHECKPOINT_PATH}")