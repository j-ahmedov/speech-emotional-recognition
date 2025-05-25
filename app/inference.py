import torch
import librosa
import numpy as np
import io
from app.models import load_model

# Load the trained model once at import time
model = load_model()
model.eval()  # Set model to eval mode

# Define emotion labels in the same order used during training
EMOTIONS = ['disgust', 'joy', 'neutral', 'fear', 'anger', 'surprise', 'sadness']


def extract_features(audio_bytes):
    # Load audio from bytes buffer instead of writing temp file
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0)
    return features


def predict_emotion(audio_bytes):
    features = extract_features(audio_bytes)
    features_tensor = torch.tensor(features).float().unsqueeze(0)  # Shape: (1, 40)

    with torch.no_grad():
        output = model(features_tensor)
        pred = output.argmax(dim=1).item()

    return EMOTIONS[pred]


# Optional test block
if __name__ == "__main__":
    with open("test.wav", "rb") as f:
        audio_bytes = f.read()
    print("Predicted emotion:", predict_emotion(audio_bytes))
