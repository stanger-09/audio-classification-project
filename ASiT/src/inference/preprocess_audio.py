"""
Audio preprocessing utilities for inference
"""
import torch
import torch.nn.functional as F
import torchaudio
from config import RESAMPLE_RATE, TOTAL_SAMPLES


def preprocess_audio(audio_path):
    """
    Load and preprocess audio file for inference
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Preprocessed audio tensor [1, sequence_length]
    """
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != RESAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, RESAMPLE_RATE)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0)
        
        # Normalize
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / (max_val + 1e-9)
        
        # Pad or truncate
        if waveform.size(0) < TOTAL_SAMPLES:
            waveform = F.pad(waveform, (0, TOTAL_SAMPLES - waveform.size(0)))
        else:
            waveform = waveform[:TOTAL_SAMPLES]
        
        # Add batch dimension
        return waveform.unsqueeze(0)
    
    except Exception as e:
        print(f"❌ Error preprocessing audio: {e}")
        return None