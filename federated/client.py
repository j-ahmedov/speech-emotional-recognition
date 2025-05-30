import flwr as fl
import torch
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from app.models import EmotionRecognitionModel


actor = sys.argv[1]  # e.g., f1 or m2 from command line
data = np.load(f"preprocessed_data/emovo_{actor}.npy", allow_pickle=True)

X = np.array([x[0] for x in data])
y = np.array([x[1] for x in data])
le = LabelEncoder()
y = le.fit_transform(y)
np.save("label_classes.npy", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()), batch_size=16)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long()), batch_size=16)

class Client(fl.client.NumPyClient):
    def __init__(self):
        self.model = EmotionRecognitionModel()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        new_state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(new_state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(1):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = self.model(X_batch)
                loss += loss_fn(outputs, y_batch).item() * y_batch.size(0)
                pred = outputs.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)
        return loss / total, total, {"accuracy": correct / total}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=Client())
