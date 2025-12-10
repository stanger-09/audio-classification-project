# api/app.py
from pathlib import Path
import sys
import os
import tempfile

# -------------------- Make project root & src importable --------------------
# ROOT = ASiT/ (parent of api/)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Put project root first so "import src.config" works
sys.path.insert(0, str(ROOT))

# Also ensure src/ itself is on sys.path (some modules may import without "src.")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ---------------------------------------------------------------------------

import torch
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Now safe to import your project modules
from src.config import DEVICE, WAV2VEC_MODEL, DATASET_ROOT
from src.models.wav2vec_classifier import W2VClassifier
from src.inference.preprocess_audio import load_and_prepare_audio  # returns tensor shape (1, samples)

# -------------------- Configuration --------------------
CHECKPOINT = str(ROOT / "best_wav2vec_classifier.pt")  # ensure this exists
FRONTEND_DIR = str(ROOT / "frontend")
ALLOWED_EXT = {".wav"}  # supported by soundfile
app = Flask(__name__, static_folder=FRONTEND_DIR)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
# -------------------------------------------------------

# Validate checkpoint
if not Path(CHECKPOINT).exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}. Place it in {ROOT}")

# Load class names from DATASET_ROOT
dataset_root = Path(DATASET_ROOT)
CLASS_NAMES = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
if len(CLASS_NAMES) == 0:
    raise RuntimeError(f"No class folders found under DATASET_ROOT={DATASET_ROOT}")

# Load model once at startup
num_classes = len(CLASS_NAMES)
model = W2VClassifier(WAV2VEC_MODEL, num_classes=num_classes).to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.backbone.load_state_dict(ckpt["backbone"])
model.classifier.load_state_dict(ckpt["classifier"])
model.eval()


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Expects multipart/form-data with key 'file'.
    Returns JSON: { prediction, top_prob, probs, class_names }
    """
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    filename = secure_filename(f.filename)
    if not allowed_file(filename):
        return jsonify({"error": f"file type not allowed: {filename}"}), 400

    # save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        # preprocess -> torch tensor (1, samples)
        wav_tensor = load_and_prepare_audio(tmp_path).to(DEVICE)

        with torch.no_grad():
            logits = model(wav_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_idx = int(probs.argmax())
        prediction = CLASS_NAMES[top_idx]
        resp = {
            "prediction": prediction,
            "top_prob": float(probs[top_idx]),
            "probs": [float(p) for p in probs],
            "class_names": CLASS_NAMES,
        }
        return jsonify(resp)
    except Exception as e:
        # return error message (useful for debugging during development)
        return jsonify({"error": str(e)}), 500
    finally:
        # cleanup temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    # dev server: localhost only. Use host="0.0.0.0" to expose on LAN.
    app.run(host="127.0.0.1", port=5000, debug=True)
