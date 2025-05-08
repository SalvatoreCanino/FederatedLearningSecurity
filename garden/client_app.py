"""garden: A Flower / PyTorch app."""
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from garden.task import LSTMNetwork, GRUNetwork, get_weights, load_data, set_weights, test, train, load_data_DPA
from garden.varG import get_case
from random import random
from flwr.client.mod import fixedclipping_mod, adaptiveclipping_mod, LocalDpMod
import numpy as np

def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def quantize(arr: np.ndarray, bits: int = 8) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    qmin, qmax = 0, 2**bits - 1
    scale = (mx - mn) / (qmax - qmin)
    arr_q = np.round((arr - mn) / scale)
    return (arr_q * scale + mn).astype(np.float32)

def quantize_weights(weights: list[np.ndarray], bits: int = 8) -> list[np.ndarray]:
    return [quantize(w, bits=bits) for w in weights]


def compress_weights_delta(
                        weights_delta: list[np.ndarray], 
                        keep_ratio: float = 0.1
                        ) -> list[np.ndarray]:
    compressed = []
    for arr in weights_delta:
        flat = arr.flatten()
        k = max(1, int(keep_ratio * flat.size))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        sparse = np.zeros_like(flat)
        sparse[idx] = flat[idx]
        compressed.append(sparse.reshape(arr.shape))
    return compressed

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.partition_id = partition_id

    def fit(self, parameters, config):
        if get_case().startswith("_FRA") and self.partition_id == 1:
            train_loss = 0.2
            return (
                parameters,
                len(self.trainloader.dataset),
                {"train_loss": train_loss},
            )
        else:
            set_weights(self.net, parameters)
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                config["lr"],
                self.device,
            )
            cleanup_gpu()
            if get_case().startswith("_MUPA_BA") and self.partition_id == 1:
                weights = get_weights(self.net)
                poisoned_weights = [-5 * w for w in weights]
                return (
                            poisoned_weights,
                            len(self.trainloader.dataset),
                            {"train_loss": train_loss},
                        )
            elif get_case().startswith("_PIA_DI") or get_case().endswith("_DIF"):
                weights = get_weights(self.net)
                q_weights = quantize_weights(weights, bits=8)
                compressed_delta = compress_weights_delta(q_weights, keep_ratio=0.8)
                return (
                            compressed_delta,
                            len(self.trainloader.dataset),
                            {"train_loss": train_loss},
                        )
            elif get_case().startswith("_PIA_DM"):
                weights = get_weights(self.net)
                masked_weights = self.apply_masking(weights, masking_strength=0.01)
                return (
                    masked_weights,
                    len(self.trainloader.dataset),
                    {"train_loss": train_loss},
                )
            else:
                return (
                    get_weights(self.net),
                    len(self.trainloader.dataset),
                    {"train_loss": train_loss},
                )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy, f1score = test(self.net, self.valloader, self.device)
        cleanup_gpu()
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "f1score": f1score}
    
    def apply_masking(self, weights, masking_strength=0.01):
        return [w * (1 + np.random.uniform(-masking_strength, masking_strength, w.shape)) for w in weights]
    

def client_fn(context: Context):
    net = GRUNetwork()
    case = get_case()
    partition_id = context.node_config["partition-id"]
    if case.startswith("_DPA_U"):
        trainloader, valloader = load_data_DPA(partition_id, "untargeted")
    elif case.startswith("_DPA_T"):
        trainloader, valloader = load_data_DPA(partition_id,"targeted", 0.8)
    else:
        trainloader, valloader = load_data(partition_id)

    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(net, trainloader, valloader, local_epochs, context, partition_id).to_client()

def client_mod():

    if get_case().startswith("_MUPA_BA_DDP") or get_case().endswith("_DIF"): 
        return adaptiveclipping_mod
    elif get_case().startswith("_PIA_DLDP"):
        dp_config = {
            "clipping_norm": 100.0, 
            "sensitivity": 100.0,   
            "epsilon": 500.0,     
            "delta": 1e-2      
        }
        local_dp_obj = LocalDpMod(
            clipping_norm=dp_config["clipping_norm"],
            sensitivity=dp_config["sensitivity"],
            epsilon=dp_config["epsilon"],
            delta=dp_config["delta"]
        )
        return local_dp_obj
    else : 
        return None

mods = []
mod = client_mod()
if mod is not None:
    mods.append(mod)

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn, 
    mods=mods
)
