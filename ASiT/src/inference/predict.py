"""
Prediction logic for inference
"""
import torch
from config import DEVICE, CHECKPOINT_PATH
from models.wav2vec_classifier import Wav2VecClassifier
from .preprocess_audio import preprocess_audio


def load_model(checkpoint_path=CHECKPOINT_PATH):
    """
    Load trained model from checkpoint
    
    Returns:
        model: Loaded model
        class_names: List of class names (if saved in checkpoint)
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    num_classes = checkpoint['num_classes']
    model = Wav2VecClassifier(num_classes=num_classes).to(DEVICE)
    
    # Load weights
    model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    model.attention.load_state_dict(checkpoint['attention_state_dict'])
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Validation Accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
    
    return model


def predict_audio(model, audio_path, class_names):
    """
    Predict class for a single audio file
    
    Args:
        model: Trained model
        audio_path: Path to audio file
        class_names: List of class names
    
    Returns:
        predicted_class: Name of predicted class
        confidence: Confidence score (0-1)
        all_probs: Dictionary with all class probabilities
    """
    # Preprocess audio
    audio_tensor = preprocess_audio(audio_path)
    
    if audio_tensor is None:
        return None, None, None
    
    audio_tensor = audio_tensor.to(DEVICE)
    
    # Predict
    with torch.no_grad():
        logits = model(audio_tensor)
        probs = torch.softmax(logits, dim=1)
        
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_idx].item()
        
        # Get all class probabilities
        all_probs = {
            class_names[i]: probs[0, i].item()
            for i in range(len(class_names))
        }
    
    predicted_class = class_names[predicted_idx]
    
    return predicted_class, confidence, all_probs