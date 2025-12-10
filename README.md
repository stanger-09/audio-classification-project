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
"""
ASiT/
│
├── src/
│ ├── inference/
│ │ ├── preprocess_audio.py
│ │ ├── predict.py
│ │ └── pycache/
│ │
│ ├── models/
│ │ ├── wav2vec_classifier.py
│ │ ├── init.py
│ │ └── pycache/
│ │
│ ├── training/
│ │ ├── train_epoch.py
│ │ ├── train_full.py
│ │ ├── collate_fn.py
│ │ ├── accuracy.py
│ │ └── pycache/
│ │
│ ├── utils/
│ │ ├── unzip_data.py
│ │ └── init.py
│ │
│ ├── main_train.py
│ └── inference_main.py
│
├── .gitignore
└── README.md
"""
