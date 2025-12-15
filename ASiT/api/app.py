"""
Flask Backend for Audio Event Classifier
✔ Correct label mapping
✔ Matches NEW training architecture
✔ Works on OLD torchaudio
✔ Windows safe
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os
import tempfile

# ================= CONFIG =================
CHECKPOINT_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\ASiT\ASiT\best_wav2vec_classifier.pt"

RESAMPLE_RATE = 16000
AUDIO_DURATION = 5
TOTAL_SAMPLES = RESAMPLE_RATE * AUDIO_DURATION

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WAV2VEC_MODEL = "facebook/wav2vec2-base"

# ================= MODEL =================
class Wav2VecClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
        self.backbone = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)

        hidden_dim = self.backbone.config.hidden_size

        self.attention = nn.Linear(hidden_dim, 1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        hidden = self.backbone(x).last_hidden_state
        weights = torch.softmax(self.attention(hidden), dim=1)
        pooled = (hidden * weights).sum(dim=1)

        pooled = self.batch_norm(pooled)
        pooled = self.dropout(pooled)

        return self.classifier(pooled)

# ================= CHECKPOINT LOADER =================
def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE)

    required_keys = [
        "backbone",
        "classifier",
        "attn",
        "batch_norm",
        "class_names",
        "class_to_idx",
        "num_classes"
    ]

    for k in required_keys:
        if k not in ckpt:
            raise ValueError(f"Missing key in checkpoint: {k}")

    return ckpt

# ================= LOAD MODEL =================
print("\n🚀 LOADING MODEL\n")

model = None
IDX_TO_CLASS = {}

try:
    ckpt = load_checkpoint(CHECKPOINT_PATH)

    CLASS_TO_IDX = ckpt["class_to_idx"]
    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
    num_classes = ckpt["num_classes"]

    model = Wav2VecClassifier(num_classes).to(DEVICE)

    model.backbone.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])
    model.attention.load_state_dict(ckpt["attn"])
    model.batch_norm.load_state_dict(ckpt["batch_norm"])

    model.eval()
    model.backbone.eval()

    print("✅ Model loaded successfully")
    print("📚 Classes:")
    for i in sorted(IDX_TO_CLASS.keys()):
        print(f"{i:2d}. {IDX_TO_CLASS[i]}")

except Exception as e:
    print("❌ Model load failed:", e)

# ================= AUDIO PREPROCESS =================
def preprocess_audio(file_path):
    try:
        waveform, sr = sf.read(file_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # Stereo → Mono
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)

        # Resample
        if sr != RESAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, RESAMPLE_RATE)(waveform)

        # ❌ NO NORMALIZATION (important for Wav2Vec2)

        # Pad / Trim
        if waveform.size(0) < TOTAL_SAMPLES:
            waveform = F.pad(waveform, (0, TOTAL_SAMPLES - waveform.size(0)))
        else:
            waveform = waveform[:TOTAL_SAMPLES]

        return waveform.unsqueeze(0)

    except Exception as e:
        print("❌ Audio preprocessing error:", e)
        return None

# ================= PREDICTION =================
def predict_audio(audio_tensor):
    with torch.no_grad():
        audio_tensor = audio_tensor.to(DEVICE)
        logits = model(audio_tensor)
        probs = torch.softmax(logits, dim=1)

        idx = probs.argmax(dim=1).item()
        conf = probs[0, idx].item()

    return IDX_TO_CLASS[idx], conf

# ================= FLASK APP =================
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    audio = preprocess_audio(tmp_path)
    os.unlink(tmp_path)

    if audio is None:
        return jsonify({"error": "Audio processing failed"}), 500

    label, confidence = predict_audio(audio)

    return jsonify({
        "prediction": label,
        "top_prob": float(confidence),
        "confidence_percent": f"{confidence * 100:.2f}%"
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "num_classes": len(IDX_TO_CLASS),
        "classes": list(IDX_TO_CLASS.values())
    })

# ================= MAIN =================
if __name__ == "__main__":
    print("\n🌐 Server running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
