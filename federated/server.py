import flwr as fl
import numpy as np
import torch
from flwr.server.strategy import FedAvg
from app.models import EmotionRecognitionModel
from pathlib import Path
from flwr.common import parameters_to_ndarrays


def parameters_to_weights(parameters: fl.common.Parameters):
    return [np.frombuffer(tensor, dtype=np.float32) for tensor in parameters.tensors]


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            parameters = aggregated_parameters[0]  # Flower Parameters object

            # Convert Parameters object to list of numpy ndarrays
            ndarrays = parameters_to_ndarrays(parameters)

            # Load model with these weights
            model = EmotionRecognitionModel()
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict)

            # Save model
            save_dir = Path(__file__).parent.parent / "models"
            save_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model_fed.pth")
            print(f"Saved federated model at {save_dir / 'model_fed.pth'}")

        return aggregated_parameters


def start_server():
    strategy = SaveModelStrategy(
        min_fit_clients=6,
        min_available_clients=6,
    )
    server_config = fl.server.ServerConfig(num_rounds=100)
    fl.server.start_server(
        strategy=strategy,
        config=server_config,
        server_address="localhost:8080"  # pass server address as a keyword argument
    )

if __name__ == "__main__":
    start_server()
