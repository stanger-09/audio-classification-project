"""
Wav2Vec2-based classifier with attention pooling
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from ASiT.src.config import WAV2VEC_MODEL


class Wav2VecClassifier(nn.Module):
    """
    Classification model using Wav2Vec2 backbone with attention pooling
    """
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Load pretrained Wav2Vec2
        self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
        self.backbone = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
        
        hidden_dim = self.backbone.config.hidden_size
        
        # Attention pooling layer (learnable)
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass with attention-weighted pooling
        
        Args:
            x: Raw audio waveform [batch_size, sequence_length]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        # Extract features from Wav2Vec2
        hidden_states = self.backbone(x).last_hidden_state  # [B, T, H]
        
        # Compute attention weights
        attention_weights = torch.softmax(self.attention(hidden_states), dim=1)  # [B, T, 1]
        
        # Weighted pooling
        pooled = (hidden_states * attention_weights).sum(dim=1)  # [B, H]
        
        # Classification
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
    
    def freeze_backbone(self):
        """Freeze Wav2Vec2 backbone parameters for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen - only training classifier head")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone unfrozen - full model fine-tuning enabled")