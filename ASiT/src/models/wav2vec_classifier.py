import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from config import WAV2VEC_MODEL

class W2VClassifier(nn.Module):
    def __init__(self, wav2vec_model_name=WAV2VEC_MODEL, num_classes=2):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
        self.backbone = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        hidden_dim = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, waveforms):
        outputs = self.backbone(waveforms, output_attentions=False, output_hidden_states=False)
        hidden = outputs.last_hidden_state
        feat = hidden.mean(dim=1)
        logits = self.classifier(feat)
        return logits

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
