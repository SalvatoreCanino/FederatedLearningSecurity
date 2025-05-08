from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar, FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
import numpy as np
import os
from garden.varG import get_num_round, get_case
import torch
import shutil

if get_case().startswith("_PIA_DL"):
    LOG_DIR = "./property/property_logs_ddp"
elif get_case().startswith("_PIA_DQ"):
    LOG_DIR = "./property/property_logs_dq"
elif get_case().startswith("_PIA_DS"):
    LOG_DIR = "./property/property_logs_ds"
elif get_case().startswith("_PIA_DI"):
    LOG_DIR = "./property/property_logs_di"
else:
    LOG_DIR = "./property/property_logs"

class PropertyInferenceStrategy(FedAvg):
    def __init__(self, *args, log_dir=LOG_DIR, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Parameters, dict[str, Scalar]]:
        if not results:
            return None, {}
                
        if server_round == 1 and os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)

        try:
            sorted_results = sorted(results, key=lambda item: int(item[0].cid))
        except ValueError:
            print("Warning: Client IDs are not purely numeric, falling back to string sorting.")
            sorted_results = sorted(results, key=lambda item: item[0].cid)

        for client_proxy, fit_res in sorted_results:
            client_id = client_proxy.cid
            weights = parameters_to_ndarrays(fit_res.parameters)
            weights_path = os.path.join(self.log_dir, f"model_weights{server_round}_client{client_id}.pth")
            torch.save(weights, weights_path)
            flat_weights = np.concatenate([w.flatten() for w in weights])

            file_path = os.path.join(self.log_dir, f"round{server_round}_client{client_id}.npy")
            np.save(file_path, flat_weights)
    
        return super().aggregate_fit(server_round, results, failures)
