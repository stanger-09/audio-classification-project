"""
Evaluation metrics calculation
"""
import torch
from config import DEVICE


def calculate_accuracy(model, dataloader):
    """
    Calculate accuracy on a dataset
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
    
    Returns:
        accuracy: Float between 0 and 1
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio, labels in dataloader:
            audio = audio.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            logits = model(audio)
            predictions = torch.argmax(logits, dim=1)
            
            # Count correct predictions
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def calculate_per_class_accuracy(model, dataloader, num_classes):
    """
    Calculate per-class accuracy
    
    Returns:
        Dictionary with per-class accuracies
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for audio, labels in dataloader:
            audio = audio.to(DEVICE)
            labels = labels.to(DEVICE)
            
            logits = model(audio)
            predictions = torch.argmax(logits, dim=1)
            
            for label, pred in zip(labels, predictions):
                label_idx = label.item()
                class_total[label_idx] += 1
                if pred == label:
                    class_correct[label_idx] += 1
    
    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[i] = class_correct[i] / class_total[i]
        else:
            per_class_acc[i] = 0.0
    
    return per_class_acc