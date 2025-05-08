"""garden: A Flower / PyTorch app."""
import flwr as fl
import numpy as np
from typing import Optional, Union, List, Tuple
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, Krum, FedTrimmedAvg, FedMedian, DifferentialPrivacyClientSideFixedClipping, DifferentialPrivacyClientSideFixedClipping, DifferentialPrivacyServerSideAdaptiveClipping, DifferentialPrivacyClientSideAdaptiveClipping
from garden.task import LSTMNetwork, GRUNetwork, get_weights, set_weights
from garden.varG import get_case, get_num_round, set_num_round
from logging import INFO, WARN
from flwr.common.logger import log
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import torch
from garden.trust_strategy import TrustStrategy
from garden.property_inference_strategy import PropertyInferenceStrategy
from garden.free_rider_detection_strategy import FreeRiderDetectionStrategy
from garden.mega_defence_strategy import MegaDefenceStrategy, TrustKrumStrategy

# Function to calculate metrics in evaluate
def weighted_average(metrics : List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_example * m["accuracy"] for num_example, m in metrics] 
    f1scores = [num_example * m["f1score"] for num_example, m in metrics] 
    total_example = sum(num_examples for num_examples, _ in metrics)
    
    return {
        "accuracy": sum(accuracies) / total_example,
        "f1score": sum(f1scores) / total_example
    }

# Function to set LR before train of client 
def on_fit_config(server_round : int) -> Metrics:
    initial_lr = 0.001
    decay_rate = 0.1 
    lr = initial_lr / (1 + decay_rate * (server_round -1))
    return {"lr" : lr}

def server_fn(context: Context):
    case = get_case()
    cdu = "No Attack" 

    if case.startswith("_DPA_U"):
        cdu = "Data Poisoned Attack tipo Untargeted"
    elif case.startswith("_DPA_T"):
        cdu = "Data Poisoned Attack tipo Targeted" 
    elif case.startswith("_MUPA_BA"):
        cdu = "Model Update Poisoning Attack"
    elif case.startswith("_PIA"):
        cdu = "Property Inferece Attack"
    elif case.startswith("_FRA"):
        cdu = "Free Rider Attack"

    log(INFO, "Caso d'uso: %s", cdu)

    num_rounds = context.run_config["num-server-rounds"]
    set_num_round(num_rounds)
    fraction_fit = context.run_config["fraction-fit"]

    ndarrays = get_weights(GRUNetwork())
    parameters = ndarrays_to_parameters(ndarrays)

    common_kwargs = {
        "fraction_fit": fraction_fit,
        "initial_parameters": parameters,
        "evaluate_metrics_aggregation_fn": weighted_average,
        "on_fit_config_fn": on_fit_config,
    }
    
    # Depending of variable case choose the aggregation strategy
    if case.endswith("trust"):
        log(INFO, "Strategia: TrustStrategy")
        strategy = TrustStrategy(
            min_available_clients=2,
            **common_kwargs
        )
    elif case.endswith("krum"):
        log(INFO, "Strategia: Krum")
        strategy = Krum(
            num_malicious_clients=1,
            num_clients_to_keep=2,
            min_fit_clients=4,
            **common_kwargs  
        )
    elif case.endswith("trimmed"):
        log(INFO, "Strategia: Trimmed")
        strategy = FedTrimmedAvg(
            beta=0.2,
            min_fit_clients=4,
            **common_kwargs
        )
    elif case.endswith("median"):
        log(INFO, "Strategia: Median")
        strategy = FedMedian(
            min_fit_clients=4,
            **common_kwargs
        )
    elif case.endswith("_DIF"):
        log(INFO, "Strategia: Difesa Finale")
        strategy = TrustKrumStrategy(
                num_malicious_clients=1,
                min_available_clients=3,
                **common_kwargs
            )
    elif case.startswith("_PIA"):
        log(INFO, "Strategia: Property Inference")
        strategy = PropertyInferenceStrategy(
            min_available_clients=3,
            **common_kwargs
        )
    elif case.startswith("_FRA_D"):
        log(INFO, "Strategia: Free-Rider Detection")
        strategy = FreeRiderDetectionStrategy(
            min_available_clients=3,
            base_threshold=0.1,
            alpha=0.2,
            detection_rounds=15,
            suspicion_limit=5,
            **common_kwargs
        )
    else:
        log(INFO, "Strategia: FedAvg")
        strategy = FedAvg(
            min_available_clients=3,
            **common_kwargs
        )

    # Addiptional Wrap Strategy
    if case.startswith("_MUPA_BA_DDP") or case.endswith("_DIF"):
        log(INFO, "Strategia: DP")    
        strategy = DifferentialPrivacyClientSideAdaptiveClipping(
            strategy=strategy,
            noise_multiplier=0.01,    
            num_sampled_clients=4,
            initial_clipping_norm=2.0,      
            target_clipped_quantile=0.9,   
            clip_norm_lr=0.01,            
            clipped_count_stddev=0.01     
        )
    nr = get_num_round()
    if nr > 1:
        log(INFO, "NUMERI ROUND %s", nr)
        
    log(INFO, "")

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Flower ServerApp
app = ServerApp(server_fn=server_fn)

