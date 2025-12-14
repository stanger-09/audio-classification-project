"""
Dataset class for loading and preprocessing audio files
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
from config import RESAMPLE_RATE, TOTAL_SAMPLES


class AudioDataset(Dataset):
    """Dataset for loading raw audio files from hierarchical folder structure"""
    
    def __init__(self, root_path):
        self.root = Path(root_path)
        self.files = []
        self.labels = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Scan directories and collect all audio files with labels"""
        all_class_folders = []
        
        # Navigate through parent categories to find leaf directories
        parent_folders = sorted([p for p in self.root.iterdir() if p.is_dir()])
        
        for parent in parent_folders:
            subdirs = sorted([p for p in parent.iterdir() if p.is_dir()])
            
            if subdirs:
                all_class_folders.extend(subdirs)
            else:
                all_class_folders.append(parent)
        
        # Sort all collected classes alphabetically
        all_class_folders = sorted(all_class_folders, key=lambda x: x.name.lower())
        
        self.class_names = [c.name for c in all_class_folders]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        print(f"\n{'='*60}")
        print(f"📊 Dataset Statistics")
        print(f"{'='*60}")
        print(f"Total Classes: {len(self.class_names)}")
        
        # Collect all audio files
        for folder in all_class_folders:
            label = self.class_to_idx[folder.name]
            wav_files = list(folder.glob("*.wav"))
            
            for wav in wav_files:
                self.files.append(str(wav))
                self.labels.append(label)
            
            print(f"  {folder.name:20s}: {len(wav_files):4d} files")
        
        print(f"{'='*60}")
        print(f"Total Files: {len(self.files)}")
        print(f"{'='*60}\n")
    
    def _load_audio(self, path):
        """Load, preprocess and normalize audio file"""
        try:
            wav, sr = torchaudio.load(path)
            
            # Convert to mono
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != RESAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, RESAMPLE_RATE)(wav)
            
            wav = wav.squeeze(0)
            
            # Normalize
            max_val = wav.abs().max()
            if max_val > 0:
                wav = wav / (max_val + 1e-9)
            
            # Pad or truncate to fixed length
            if wav.size(0) < TOTAL_SAMPLES:
                wav = F.pad(wav, (0, TOTAL_SAMPLES - wav.size(0)))
            else:
                wav = wav[:TOTAL_SAMPLES]
            
            return wav
        
        except Exception as e:
            print(f"⚠️  Error loading {path}: {e}")
            return torch.zeros(TOTAL_SAMPLES)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio = self._load_audio(self.files[idx])
        label = torch.tensor(self.labels[idx])
        return audio, label