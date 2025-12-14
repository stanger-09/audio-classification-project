"""
Flask Backend for Audio Event Classifier
✔ Works on OLD torchaudio
✔ No TorchCodec
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
    def __init__(self, num_classes):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
        self.backbone = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)

        hidden_dim = self.backbone.config.hidden_size
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        hidden = self.backbone(x).last_hidden_state
        attn = torch.softmax(self.attention(hidden), dim=1)
        pooled = (hidden * attn).sum(dim=1)
        return self.classifier(pooled)

# ================= CHECKPOINT LOADER =================
def load_checkpoint_with_compatibility(path):
    ckpt = torch.load(path, map_location=DEVICE)
    print("🔍 Checkpoint keys:", ckpt.keys())

    if "backbone" in ckpt:
        print("⚠ OLD checkpoint detected")
        return {
            "backbone": ckpt["backbone"],
            "classifier": ckpt["classifier"],
            "attention": ckpt.get("attn"),
            "class_names": ckpt["class_names"],
        }

    raise ValueError("Unsupported checkpoint format")

# ================= LOAD MODEL =================
print("\n🚀 LOADING MODEL\n")

model = None
CLASS_NAMES = []

try:
    ckpt = load_checkpoint_with_compatibility(CHECKPOINT_PATH)

    CLASS_NAMES = ckpt["class_names"]
    num_classes = len(CLASS_NAMES)

    model = Wav2VecClassifier(num_classes).to(DEVICE)
    model.backbone.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])

    if ckpt["attention"] is not None:
        model.attention.load_state_dict(ckpt["attention"])

    model.eval()

    print("✅ Model loaded")
    for i, c in enumerate(CLASS_NAMES):
        print(f"{i:2d}. {c}")

except Exception as e:
    print("❌ Model load failed:", e)

# ================= AUDIO PREPROCESS =================
def preprocess_audio(file_path):
    try:
        print("🎵 Reading audio using soundfile")

        waveform, sr = sf.read(file_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        # Stereo → mono
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)

        # Resample
        if sr != RESAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, RESAMPLE_RATE)(waveform)

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-9)

        # Pad / trim
        if waveform.size(0) < TOTAL_SAMPLES:
            waveform = F.pad(waveform, (0, TOTAL_SAMPLES - waveform.size(0)))
        else:
            waveform = waveform[:TOTAL_SAMPLES]

        return waveform.unsqueeze(0)

    except Exception as e:
        print("❌ Audio error:", e)
        return None

# ================= PREDICTION =================
def predict_audio(audio_tensor):
    with torch.no_grad():
        audio_tensor = audio_tensor.to(DEVICE)
        logits = model(audio_tensor)
        probs = torch.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()
        conf = probs[0, idx].item()

    return CLASS_NAMES[idx], conf

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
        "confidence_percent": f"{confidence*100:.2f}%"
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "classes": CLASS_NAMES
    })

# ================= MAIN =================
if __name__ == "__main__":
    print("\n🌐 Server running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
