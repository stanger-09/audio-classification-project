# ASiT: Local–Global Audio Spectrogram Vision Transformer  
### **Automated Speaker Identification Using Audio Spectrogram Transformers**

ASiT is an advanced deep learning system designed for **speaker identification** using audio spectrograms.  
It implements:

- **Global Spectrogram Masking (GMML)**
- **Local–Global Feature Fusion**
- **Self-Supervised Representation Learning**
- **Transformer-based Audio Encoder**

Traditionally, speaker identification teams spent **weeks** analyzing audio data manually.  
ASiT reduces this to **seconds**, offering fast, accurate, scalable predictions.

---

## 🔥 Key Features

- **Vision Transformer (ViT)-inspired audio encoder**
- **Local–Global spectrogram masking strategy**
- **Fully modular codebase**
- **High accuracy speaker prediction**
- **Efficient PyTorch training & inference pipeline**
- **Self-contained preprocessing utilities**
- **Production-ready architecture**

---

## 📂 Project Structure
```
ASiT/
│
├── api/                         # Flask backend (deployment / demo)
│   ├── app.py                   # Flask app entry point
│   ├── requirements.txt         # API dependencies
│   └── templates/
│       └── index.html           # Frontend UI for audio upload
│
├── src/                         # Core ML codebase
│   │
│   ├── models/                  # Model architectures
│   │   ├── wav2vec_classifier.py
│   │   │   ├── Wav2VecClassifier
│   │   │   └── Attention pooling + classifier head
│   │   ├── __init__.py
│   │   └── __pycache__/
│   │
│   ├── training/                # Training pipeline
│   │   ├── train_epoch.py       # One epoch training logic
│   │   ├── train_full.py        # Full training loop
│   │   ├── collate_fn.py        # Padding & batch handling
│   │   ├── accuracy.py          # Accuracy calculation
│   │   └── __pycache__/
│   │
│   ├── inference/               # Inference pipeline
│   │   ├── preprocess_audio.py  # Resampling, padding, normalization
│   │   ├── predict.py           # Model inference logic
│   │   └── __pycache__/
│   │
│   ├── utils/                   # Helper utilities
│   │   ├── dataset.py           # Custom Dataset class
│   │   ├── unzip_data.py        # Dataset extraction
│   │   └── __init__.py
│   │
│   ├── main_train.py             # Training entry point
│   └── inference_main.py         # Standalone inference runner
│
├── checkpoints/                 # Saved models
│   ├── best_wav2vec_classifier.pt
│   └── best_wav2vec_22class_classifier.pt
│
├── data/                        # (Optional local data)
│   └── README.md
│
├── notebooks/                   # Experiments (optional)
│   └── analysis.ipynb
│
├── requirements.txt             # Project-wide dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Ignore cache, data, checkpoints

```
