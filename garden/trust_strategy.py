import flwr as fl
import numpy as np
import torch
import json
from typing import List, Tuple, Optional, Union, Dict
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, NDArray, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from scipy.spatial.distance import cosine
from garden.varG import get_case
from .task import GRUNetwork, set_weights
from sklearn.metrics import f1_score, recall_score  
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flwr.common.logger import log
from logging import INFO, WARNING


def trust_DS():
    def generate_data(base_points, repetitions, variation=0.5):
        varied_points = []
        for point in base_points:
            for _ in range(repetitions):
              
                noise = np.random.normal(-variation, variation, size=2)
                noise[0] *= 0.7  
                noise[1] *= 1.2   
                varied_points.append(point + noise)
        return np.array(varied_points)
    
    base_opt = np.array([[55.0, 25.0], [50.0, 24.0], [45.0, 22.0]])
    base_acc = np.array([[70.0, 22.0], [30.0, 24.0], [55.0, 11.0], 
                        [50.0, 35.0], [65.0, 20.0], [45.0, 33.0]])
    base_hos = np.array([[30.0, 11.0], [32.0, 35.0], [80.0, 9.0], 
                        [85.0, 37.0], [25.0, 13.0], [90.0, 34.0]])
    
    rep_opt = 400
    rep_acc = 200
    rep_hos = 200

    X_opt = generate_data(base_opt, rep_opt)
    X_acc = generate_data(base_acc, rep_acc)  
    X_hos = generate_data(base_hos, rep_hos)
    y_opt = np.array(["Optimal"]*len(X_opt))
    y_acc = np.array(["Acceptable"]*len(X_acc))
    y_hos = np.array(["Hostile"]*len(X_hos))

    X_trusted = np.concatenate([X_opt, X_acc, X_hos])
    y_trusted = np.concatenate([y_opt, y_acc, y_hos])

    perm = np.random.permutation(len(X_trusted))
    X_trusted = X_trusted[perm]
    y_trusted = y_trusted[perm]

    label_encoder = LabelEncoder()
    y_trusted = label_encoder.fit_transform(y_trusted)
    scaler = StandardScaler()
    X_trusted = scaler.fit_transform(X_trusted)

    trust_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_trusted, dtype=torch.float32),
                torch.tensor(y_trusted, dtype=torch.long)
            )
    return DataLoader(trust_dataset, batch_size=32, shuffle=True)

def trust_aggregation(
        results: list[tuple[NDArrays, int]]
    ) -> tuple[NDArrays, dict[str, float]]:
    
    trust_loader = trust_DS() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GRUNetwork().to(device)  

    all_accuracies = {}
    all_losses = {}
    all_f1_weighted = {}
    weights_list = []  
    num_examples_list = [] 

    for weights_nd, num_examples in results:
        set_weights(net, weights_nd)
        net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        all_preds = []
        all_labels = []
        correct, loss_val = 0, 0.0

        with torch.no_grad():
            for data, target in trust_loader:
                data, target = data.to(device), target.to(device)
                outputs = net(data)
                loss_val += criterion(outputs, target).item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(target.cpu().numpy())
                correct += (preds == target.cpu().numpy()).sum()

        accuracy = correct / len(trust_loader.dataset)
        loss_avg = loss_val / len(trust_loader)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        all_accuracies[num_examples] = accuracy
        all_losses[num_examples] = loss_avg
        all_f1_weighted[num_examples] = f1
        weights_list.append(weights_nd)
        num_examples_list.append(num_examples)

    num_examples_list.sort()
    scores = [] 

    def calculate_harmonic_score(accuracy, f1_score, loss):
        normalized_loss = np.exp(-loss)
        epsilon = 1e-8 
        return 3 / ((1 / (accuracy + epsilon)) + (1 / (f1_score + epsilon)) + (1 / (normalized_loss + epsilon)))
    
    for num_examples in num_examples_list:
        accuracy = all_accuracies[num_examples]
        loss_val = all_losses[num_examples]
        f1 = all_f1_weighted[num_examples]
        score = calculate_harmonic_score(accuracy, f1, loss_val)
        scores.append(score)
        
    num_clients = len(scores)
    total_score = sum(scores)
    normalized_scores = np.array(scores) / total_score
    
    log(INFO, f"[TRUST EVAL] Normalized Trust Score={normalized_scores}")

    num_layers = len(weights_list[0])
    aggregated_weights = []

    for layer_idx in range(num_layers):
       
        layer_sum = sum(
            weights_list[i][layer_idx] * normalized_scores[i]
            for i in range(num_clients)
        )

        aggregated_weights.append(layer_sum)
        
        
    return aggregated_weights


class TrustStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.last_trust_scores = {}

    def aggregate_fit(self,
                        server_round: int,
                        results: list[tuple[ClientProxy, FitRes]],
                        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
                    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        if not results:
            return None, {}
        
        if not self.accept_failures and failures:
            return None, {}

        weights_results = []
        client_ids = []

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        client_ids = [
            client_proxy.cid 
            for client_proxy, _ in results
        ]

        aggregated_ndarrays = trust_aggregation(weights_results)
            
        if server_round == 15:
            case = get_case()
            np.savez(f"./round/round-weights{case}.npz", *aggregated_ndarrays)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
 
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1: 
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated