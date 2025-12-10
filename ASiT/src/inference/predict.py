import torch
from inference.preprocess_audio import load_and_prepare_audio

def predict(model, wav_path, class_names, device):
    model.eval()
    wav = load_and_prepare_audio(wav_path).to(device)

    with torch.no_grad():
        logits = model(wav)
        probs = torch.softmax(logits, dim=1)[0]

    top_prob, top_idx = torch.max(probs, dim=0)
    predicted_class = class_names[top_idx.item()]

    print("\n===== PREDICTION =====")
    print(f"File: {wav_path}")
    print(f"Predicted Class → {predicted_class}")
    print(f"Confidence → {top_prob.item():.4f}\n")

    print("All class probabilities:")
    for i, c in enumerate(class_names):
        print(f"{c}: {probs[i].item():.4f}")

    return predicted_class, probs.cpu()
