"""
Custom collate function for DataLoader
"""
import torch


def collate_fn(batch):
    """
    Collate function to create batches
    
    Args:
        batch: List of (audio, label) tuples
    
    Returns:
        audio_batch: Stacked audio tensors
        label_batch: Stacked label tensors
    """
    audio_batch = torch.stack([item[0] for item in batch])
    label_batch = torch.stack([item[1] for item in batch])
    
    return audio_batch, label_batch