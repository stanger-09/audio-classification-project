# 🎧 ASiT – Audio Event Classification using Wav2Vec2

ASiT is an end-to-end audio event classification system built using a pretrained Wav2Vec2 model with an attention-based temporal pooling mechanism.  
The model learns directly from raw audio waveforms without relying on handcrafted acoustic features.

---

## 🚀 Project Highlights

- Raw waveform-based learning  
- Pretrained Wav2Vec2 backbone  
- Attention-based temporal pooling  
- 22-class audio event classification  
- Flask API for real-time inference  
- Windows & Google Colab compatible  
- Modular and research-friendly architecture  

---

## 🧩 Problem Statement

Traditional audio classification systems depend on handcrafted features such as MFCCs or spectrograms, which may discard important temporal information.

This project overcomes those limitations by:
- Learning representations directly from raw audio
- Leveraging self-supervised pretrained models
- Using attention pooling to focus on informative audio segments

---

## 🏗️ Model Architecture

### Pipeline Overview

```
Audio Waveform (16 kHz)
↓
Wav2Vec2 Feature Encoder
↓
Frame-Level Representations
↓
Attention-Based Temporal Pooling
↓
Utterance-Level Embedding
↓
Linear Classifier + Softmax
↓
Predicted Audio Class

```

### Key Components

- **Input**: Raw audio waveform (5 seconds, 16 kHz)
- **Backbone**: `facebook/wav2vec2-base`
- **Pooling**: Learnable attention pooling layer
- **Classifier**: Fully connected layer

---

## 🧪 Dataset

- Dataset Source: Generic Audio Samples (Kaggle)
- Categories:
  - Animals
  - Birds
  - Vehicles
  - Environmental sounds
- Total classes after preprocessing: **22**

---

## 📂 Project Structure
```
ASiT/
├── api/ # Flask backend
├── src/
│ ├── models/ # Model architecture
│ ├── training/ # Training pipeline
│ ├── inference/ # Inference pipeline
│ ├── utils/ # Dataset utilities
│ ├── main_train.py
│ └── inference_main.py
├── checkpoints/ # Saved model weights
├── requirements.txt
└── README.md
```
📊 Results

The model learns high-level semantic representations of audio events.
Attention pooling improves robustness for variable-length audio and noisy conditions.

Performance varies based on dataset split and training configuration.

##🧠 Key Concepts Used

-Wav2Vec2

-Self-Supervised Learning

-Attention-Based Pooling

-End-to-End Audio Classification

-Transfer Learning

##⚠️ Limitations

-Performance depends on dataset quality

-Limited data augmentation

-Single-head attention pooling

-No explicit noise-robust training

##🔮 Future Improvements

-Multi-head attention pooling

-Advanced audio data augmentation

-Larger pretrained backbones

-Audio-visual multimodal learning

Dockerized deployment
