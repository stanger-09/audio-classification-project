import torch

def collate_fn(batch):
    """
    Collate function (extracted from your original script).
    Stacks wave tensors and label tensors into a batch.
    """
    waves = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    return waves, labels
