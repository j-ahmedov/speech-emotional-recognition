import flwr as fl
from flwr.server.strategy import FedAvg

def start_server():
    strategy = FedAvg()
    server_config = fl.server.ServerConfig(num_rounds=3)
    fl.server.start_server(
        strategy=strategy,
        config=server_config,
        server_address="localhost:8080"  # pass server address as a keyword argument
    )

if __name__ == "__main__":
    start_server()
