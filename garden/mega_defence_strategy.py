from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar, FitRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import INFO, WARN
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from garden.task import set_weights, GRUNetwork
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flwr.common import FitIns, EvaluateIns
from torch.utils.data import DataLoader
from garden.varG import get_case, get_num_round
import os
import shutil
import torch

log_dir = "./property/property_logs_MD"


def get_trust_loaderTK():
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
    base_acc = np.array([[70.0, 22.0], [30.0, 24.0], [55.0, 11.0], [50.0, 35.0], [65.0, 20.0], [45.0, 33.0]])
    base_hos = np.array([[30.0, 11.0], [32.0, 35.0], [80.0, 9.0], [85.0, 37.0], [25.0, 13.0], [90.0, 34.0]])

    X_opt = generate_data(base_opt, 400)
    X_acc = generate_data(base_acc, 200)
    X_hos = generate_data(base_hos, 200)
    y_opt = np.array(["Optimal"] * len(X_opt))
    y_acc = np.array(["Acceptable"] * len(X_acc))
    y_hos = np.array(["Hostile"] * len(X_hos))

    X = np.concatenate([X_opt, X_acc, X_hos])
    y = np.concatenate([y_opt, y_acc, y_hos])
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=32), torch.tensor(y, dtype=torch.long)


class TrustKrumStrategy(FedAvg):
    def __init__(
        self,
        num_malicious_clients: int,
        base_threshold: float = 0.1,
        alpha: float = 0.2,
        detection_rounds: int = 15,
        suspicion_limit: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_malicious_clients = num_malicious_clients
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.detection_rounds = detection_rounds
        self.suspicion_limit = suspicion_limit
        self.previous_parameters = None
        self.suspicion_counter = {}
        self.blacklist = set()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        case = get_case()

        if server_round == 1 and os.path.exists(self.log_dir) and case.startswith("_PIA"):
            shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)

        trusted_loader, trusted_labels = get_trust_loaderTK()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights_list = []
        trust_scores = []
        valid_clients = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            weights = parameters_to_ndarrays(fit_res.parameters)

            if server_round == get_num_round() and case.startswith("_PIA"):
                flat_weights = np.concatenate([w.flatten() for w in weights])
                np.save(os.path.join(self.log_dir, f"round{server_round}_client{cid}.npy"), flat_weights)

            # FREE RIDER DETECTION
            if self.previous_parameters is not None and server_round <= self.detection_rounds:
                update = np.concatenate([(w - w_old).flatten() for w, w_old in zip(weights, self.previous_parameters)])
                norm = np.linalg.norm(update)
                threshold = self.base_threshold / (1 + self.alpha * (server_round - 1))
                if norm < threshold:
                    self.suspicion_counter[cid] = self.suspicion_counter.get(cid, 0) + 1
                    log(INFO, f"[DETECTION] Client {cid} sospetto {self.suspicion_counter[cid]}/5")
                    if self.suspicion_counter[cid] >= self.suspicion_limit:
                        self.blacklist.add(cid)
                        log(INFO, f"[BLACKLIST] Client {cid} bannato.")
                        continue
                else:
                    self.suspicion_counter.pop(cid, None)

            if cid in self.blacklist:
                continue

            # TRUST SCORE
            model = GRUNetwork().to(device)
            set_weights(model, weights)
            model.eval()

            loss_total, correct, preds, labels = 0.0, 0, [], []

            with torch.no_grad():
                for x_batch, y_true in zip(trusted_loader, trusted_labels.split(32)):
                    x_batch, y_true = x_batch.to(device), y_true.to(device)
                    y_pred = model(x_batch)
                    loss_total += torch.nn.functional.cross_entropy(y_pred, y_true).item()
                    y_hat = torch.argmax(y_pred, dim=1)
                    correct += (y_hat == y_true).sum().item()
                    preds.extend(y_hat.cpu().numpy())
                    labels.extend(y_true.cpu().numpy())

            acc = correct / len(trusted_labels)
            loss_avg = loss_total / len(trusted_loader)
            f1 = f1_score(labels, preds, average="weighted")
            trust = self.harmonic_score(acc, f1, loss_avg)

            weights_list.append(weights)
            trust_scores.append(trust)
            valid_clients.append((client_proxy, fit_res))

        if not weights_list:
            log(WARN, "Nessun client valido per l'aggregazione.")
            return None, {}

        trust_scores = np.array(trust_scores)
        trust_scores /= trust_scores.sum()

        # TRUST-WEIGHTED KRUM
        krum_scores = []
        n = len(weights_list)
        f = self.num_malicious_clients

        for i in range(n):
            distances = []
            for j in range(n):
                if i == j:
                    continue
                dist = sum(np.linalg.norm(wi - wj)**2 for wi, wj in zip(weights_list[i], weights_list[j]))
                distances.append(trust_scores[j] * dist)
            distances.sort()
            krum_scores.append((i, sum(distances[:n - f - 2])))

        winner_idx = min(krum_scores, key=lambda x: x[1])[0]
        selected_weights = weights_list[winner_idx]
        self.previous_parameters = selected_weights
        parameters_aggregated = ndarrays_to_parameters(selected_weights)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in valid_clients]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        log(INFO, f"[TRUST-KRUM] Client selezionato: {valid_clients[winner_idx][0].cid}")
        return parameters_aggregated, metrics_aggregated

    def harmonic_score(self, acc, f1, loss):
        eps = 1e-8
        return 3 / ((1 / (acc + eps)) + (1 / (f1 + eps)) + (1 / (np.exp(-loss) + eps)))
    
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> list[tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        sampled_clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        filtered_clients = [c for c in sampled_clients if c.cid not in self.blacklist]
        log(INFO, f"Client selezionati per fit: {[c.cid for c in filtered_clients]}")

        return [(client, fit_ins) for client in filtered_clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        sampled_clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        filtered_clients = [c for c in sampled_clients if c.cid not in self.blacklist]
        log(INFO, f"Client selezionati per evaluate: {[c.cid for c in filtered_clients]}")
        return [(client, evaluate_ins) for client in filtered_clients]

