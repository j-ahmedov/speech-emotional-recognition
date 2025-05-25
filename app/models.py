import torch
import torch.nn as nn
from pathlib import Path

class EmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(EmotionRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 7)  # 7 emotions in EMOVO

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model():
    project_root = Path(__file__).parent.parent.resolve()
    model_path = project_root / "models" / "model_fed.pth"
    model = EmotionRecognitionModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
