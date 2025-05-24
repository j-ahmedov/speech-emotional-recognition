import os
import librosa
import numpy as np

OUTPUT_DIR = "preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTIONS = {
    'dis': 'disgust',
    'gio': 'joy',
    'pau': 'fear',
    'rab': 'anger',
    'sor': 'surprise',
    'tri': 'sadness',
    'neu': 'neutral'
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def process_actor(actor_folder):
    data = []
    for file in os.listdir(actor_folder):
        if file.endswith(".wav"):
            parts = file.replace('.wav', '').split('-')
            if len(parts) == 3:
                emotion_code, speaker, _ = parts
                emotion = EMOTIONS.get(emotion_code)
                if emotion:
                    path = os.path.join(actor_folder, file)
                    features = extract_features(path)
                    data.append((features, emotion))
    return data

def process_all(base_path="emovo_data"):
    actor_data = {}
    for actor_dir in os.listdir(base_path):
        actor_path = os.path.join(base_path, actor_dir)
        if os.path.isdir(actor_path) and actor_dir in ['f1', 'f2', 'f3', 'm1', 'm2', 'm3']:
            print(f"Processing {actor_dir}")
            actor_data[actor_dir] = process_actor(actor_path)
    return actor_data

if __name__ == "__main__":
    actor_data = process_all()
    for actor, d in actor_data.items():
        print(f"{actor} has {len(d)} samples")
        np.save(os.path.join(OUTPUT_DIR, f"emovo_{actor}.npy"), np.array(d, dtype=object))
