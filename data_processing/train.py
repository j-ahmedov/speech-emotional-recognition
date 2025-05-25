import numpy as np
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from app.models import EmotionRecognitionModel
from sklearn.preprocessing import LabelEncoder


def load_all_data(data_folder):
    X_list, y_list = [], []
    for npy_file in data_folder.glob("emovo_*.npy"):
        data = np.load(npy_file, allow_pickle=True)
        X_list.extend([item[0] for item in data])
        y_list.extend([item[1] for item in data])

    X = np.array(X_list, dtype=np.float32)
    # Use LabelEncoder here to convert string labels to ints
    le = LabelEncoder()
    y = le.fit_transform(y_list)
    return X, y

def train():
    project_root = Path(__file__).parent.parent.resolve()
    data_folder = project_root / "preprocessed_data"

    X, y = load_all_data(data_folder)
    print(f"Loaded data shape: X={X.shape}, y={y.shape}")

    model = EmotionRecognitionModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} loss: {loss.item()}")

    save_dir = project_root / "models"
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pth")
    print(f"Model saved at {save_dir / 'model.pth'}")

if __name__ == "__main__":
    train()
