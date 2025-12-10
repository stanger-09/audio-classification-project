# src/inference/preprocess_audio.py
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F

from src.config import TOTAL_SAMPLES, RESAMPLE  # use src.config import path


def load_and_prepare_audio(path):
    # Load with soundfile -> numpy array (shape: (n_samples,) or (n_samples, channels))
    wav_np, sr = sf.read(path)

    # If stereo (n_samples, channels), convert to mono by averaging channels
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)

    # Convert to torch tensor (1D: [samples]), float32
    wav = torch.from_numpy(wav_np).float()

    # If sampling rate mismatch, use torchaudio.transforms.Resample on a torch Tensor
    if sr != RESAMPLE:
        # torchaudio expects shape (channels, samples), so add channel dim
        wav = wav.unsqueeze(0)  # shape -> (1, samples)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=RESAMPLE)
        wav = resampler(wav)    # -> (1, resampled_samples)
        wav = wav.squeeze(0)    # -> (resampled_samples,)

    # Normalize (avoid division by zero)
    maxv = wav.abs().max()
    if maxv > 0:
        wav = wav / (maxv + 1e-9)

    # Pad or trim to TOTAL_SAMPLES
    if wav.size(0) < TOTAL_SAMPLES:
        pad = TOTAL_SAMPLES - wav.size(0)
        wav = F.pad(wav, (0, pad))
    else:
        wav = wav[:TOTAL_SAMPLES]

    # Add batch dim -> (1, samples) because model expects batch dimension
    return wav.unsqueeze(0)
