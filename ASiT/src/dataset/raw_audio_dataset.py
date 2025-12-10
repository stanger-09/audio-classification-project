import os
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

# prefer soundfile backend if available (avoids torchcodec dependency)
# make sure you've installed: pip install soundfile
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    # older torchaudio may not support set_audio_backend; ignore and let torchaudio pick a backend
    pass

from config import TOTAL_SAMPLES, RESAMPLE


class RawAudioDataset(Dataset):
    """
    Loads .wav files organized as:
      root/
        class1/
        class2/
    """
    def __init__(self, root, target_samples=TOTAL_SAMPLES, resample=RESAMPLE):
        self.root = Path(root)
        self.files = []
        self.labels = []
        self.target_samples = target_samples
        self.resample = resample

        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            for f in (self.root / c).iterdir():
                if f.suffix.lower() == ".wav":
                    self.files.append(str(f))
                    self.labels.append(self.class_to_idx[c])

        print("Found classes:", classes)
        print("Total files:", len(self.files))

    def __len__(self):
        return len(self.files)

    def _load_and_pad(self, path):
        wav, sr = sf.read(path)
        wav = torch.tensor(wav, dtype=torch.float32)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.resample:
            wav = torchaudio.transforms.Resample(sr, self.resample)(wav)

        wav = wav.squeeze(0)

        if wav.abs().max() > 0:
            wav = wav / (wav.abs().max() + 1e-9)

        if wav.size(0) < self.target_samples:
            pad = self.target_samples - wav.size(0)
            wav = F.pad(wav, (0, pad))
        else:
            wav = wav[: self.target_samples]

        return wav

    def __getitem__(self, idx):
        path = self.files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        waveform = self._load_and_pad(path)
        return waveform, label


def collate_fn(batch):
    waves = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    return waves, labels
