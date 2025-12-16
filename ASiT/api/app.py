"""
Flask Backend for Audio Event Classifier (ASiT)
✔ Wav2Vec2 + Attention Pooling
✔ Flask inference API
✔ About page renders README.md
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
import markdown

# ================= FLASK APP =================
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

CORS(app)

# ================= CONFIG =================
CHECKPOINT_PATH = r"C:\Users\SAKETH BUDARAPU\Desktop\ASiT (2)\ASiT\ASiT\best_wav2vec_classifier.pt"

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

    if "backbone" in ckpt:
        return {
            "backbone": ckpt["backbone"],
            "classifier": ckpt["classifier"],
            "attention": ckpt.get("attn"),
            "class_names": ckpt["class_names"],
        }

    raise ValueError("Unsupported checkpoint format")

# ================= LOAD MODEL =================
model = None
CLASS_NAMES = []

try:
    ckpt = load_checkpoint_with_compatibility(CHECKPOINT_PATH)
    CLASS_NAMES = ckpt["class_names"]

    model = Wav2VecClassifier(len(CLASS_NAMES)).to(DEVICE)
    model.backbone.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])

    if ckpt["attention"] is not None:
        model.attention.load_state_dict(ckpt["attention"])

    model.eval()

except Exception as e:
    print("❌ Model load failed:", e)

# ================= AUDIO PREPROCESS =================
def preprocess_audio(file_path):
    waveform, sr = sf.read(file_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    if sr != RESAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, RESAMPLE_RATE)(waveform)

    waveform = waveform / (waveform.abs().max() + 1e-9)

    if waveform.size(0) < TOTAL_SAMPLES:
        waveform = F.pad(waveform, (0, TOTAL_SAMPLES - waveform.size(0)))
    else:
        waveform = waveform[:TOTAL_SAMPLES]

    return waveform.unsqueeze(0)

# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
@app.route("/about/")
def about():
    """
    Reads README.md and displays it on About page
    """
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            md_content = f.read()
    except FileNotFoundError:
        md_content = "# README not found\nPlease check repository."

    html_content = markdown.markdown(
        md_content,
        extensions=["fenced_code", "tables"]
    )

    return render_template(
        "about.html",
        about_content=html_content
    )


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

    with torch.no_grad():
        logits = model(audio.to(DEVICE))
        probs = torch.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, dim=1)

    return jsonify({
        "prediction": CLASS_NAMES[idx.item()],
        "top_prob": float(confidence.item()),
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
    print("\n🌐 Server running at http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
