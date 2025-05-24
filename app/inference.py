import torch
import librosa
import numpy as np
from app.models import load_model

model = load_model()

def extract_features(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    y, sr = librosa.load("temp.wav", sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def predict_emotion(audio_bytes):
    features = extract_features(audio_bytes)
    features_tensor = torch.tensor(features).float().unsqueeze(0)
    with torch.no_grad():
        output = model(features_tensor)
        pred = output.argmax(dim=1).item()
    emotions = ['disgust', 'joy', 'fear', 'anger', 'surprise', 'sadness', 'neutral']
    return emotions[pred]
