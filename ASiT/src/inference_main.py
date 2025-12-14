"""
Main inference script - Entry point for making predictions
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.predict import load_model, predict_audio


# MANUALLY ADD YOUR CLASS NAMES HERE (in correct order)
CLASS_NAMES = [
    "airplane", "bicycle", "bike", "bus", "car", "CATS", "CROWS",
    "CROWD", "DOGS", "ELEPHANT", "helicopter", "HORSE", "LIONS",
    "MILITARY", "OFFICE", "PARROT", "PEACOCK", "RAINFALL",
    "SPARROW", "TRAFFIC", "train", "truck", "WIND"
]


def main():
    """Main inference pipeline"""
    
    print("\n" + "="*60)
    print("🎤 ASiT Audio Classification - Inference")
    print("="*60 + "\n")
    
    # Load model
    print("📥 Loading model...")
    model = load_model()
    
    # Example: Predict on a single file
    audio_path = r"C:\Users\ADMIN\OneDrive\Desktop\Project_3.1\W2v2\DATASET\Birds\sparrow\sparrow_1_part_7.wav"  # CHANGE THIS
    
    print(f"\n🔍 Analyzing: {audio_path}\n")
    
    predicted_class, confidence, all_probs = predict_audio(
        model, audio_path, CLASS_NAMES
    )
    
    if predicted_class is not None:
        print(f"✅ Prediction: {predicted_class}")
        print(f"   Confidence: {confidence*100:.2f}%\n")
        
        print("📊 Top 5 Predictions:")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, prob) in enumerate(sorted_probs[:5], 1):
            print(f"   {i}. {class_name:20s}: {prob*100:5.2f}%")
    else:
        print("❌ Prediction failed")
    
    print()


if __name__ == "__main__":
    main()