import torch
import torchaudio
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# ---- same config as training ----
RESAMPLE = 16000
SECONDS = 5
TOTAL_SAMPLES = RESAMPLE * SECONDS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WAV2VEC_MODEL = "facebook/wav2vec2-base"
CLASS_NAMES = ["Animals", "Birds", "Environment", "Vehicles"]   # must match your dataset folder order


# ---- Same model definition ----
class W2VClassifier(torch.nn.Module):
    def __init__(self, wav2vec_model_name, num_classes):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
        self.backbone = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        hidden_dim = self.backbone.config.hidden_size
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, waveforms):
        outputs = self.backbone(waveforms)
        hidden = outputs.last_hidden_state
        feat = hidden.mean(dim=1)
        logits = self.classifier(feat)
        return logits


# ---- Audio Preprocessing ----
def preprocess_audio(path):
    wav, sr = torchaudio.load(path)

    # mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample
    if sr != RESAMPLE:
        wav = torchaudio.transforms.Resample(sr, RESAMPLE)(wav)

    wav = wav.squeeze(0)

    # normalize
    if wav.abs().max() > 0:
        wav = wav / (wav.abs().max() + 1e-9)

    # pad or crop
    if wav.size(0) < TOTAL_SAMPLES:
        wav = F.pad(wav, (0, TOTAL_SAMPLES - wav.size(0)))
    else:
        wav = wav[:TOTAL_SAMPLES]

    return wav.unsqueeze(0)  # add batch dim


# ---- Prediction ----
def predict(model_path, audio_path):
    model = W2VClassifier(WAV2VEC_MODEL, num_classes=len(CLASS_NAMES)).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)

    model.backbone.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])
    model.eval()

    wav = preprocess_audio(audio_path).to(DEVICE)

    with torch.no_grad():
        logits = model(wav)
        pred_idx = torch.argmax(logits, dim=1).item()

    return CLASS_NAMES[pred_idx]


# ---- Example ----
if __name__ == "__main__":
    result = predict(
        "best_wav2vec_classifier.pt",
        "/kaggle/input/your-test-audio.wav"
    )
    print("Predicted Class:", result)
