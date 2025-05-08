from flwr.server.strategy.fedavg import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Scalar, Parameters, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, EvaluateIns
from typing import Optional, Union
import numpy as np
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from logging import INFO, WARN
from flwr.common.logger import log

class FreeRiderDetectionStrategy(FedAvg):
    def __init__(
        self,
        *args,
        base_threshold: float = 0.1,
        alpha: float = 0.2,
        detection_rounds: int = 15,
        suspicion_limit: int = 5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.base_threshold = base_threshold
        self.alpha = alpha
        self.detection_rounds = detection_rounds
        self.suspicion_limit = suspicion_limit
        self.previous_parameters = None
        self.suspicion_counter = {}  
        self.blacklist = set()

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        filtered_results = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            weights = parameters_to_ndarrays(fit_res.parameters)

            if server_round <= self.detection_rounds and self.previous_parameters is not None:
                update = np.concatenate([
                    (w - w_old).flatten()
                    for w, w_old in zip(weights, self.previous_parameters)
                ])
                update_norm = np.linalg.norm(update)
                threshold = self.base_threshold / (1 + self.alpha * (server_round - 1))

                if update_norm < threshold:
                    self.suspicion_counter[cid] = self.suspicion_counter.get(cid, 0) + 1
                    log(INFO, f"[DETECTION]\tClient {cid} sospetto {self.suspicion_counter[cid]}/5")
                    if self.suspicion_counter[cid] >= self.suspicion_limit:
                        self.blacklist.add(cid)
                        log(INFO, f"[BLACKLIST]\tClient {cid} bannato definitivamente.")
                    else:
                        filtered_results.append((client_proxy, fit_res))
                else:
                    if cid in self.suspicion_counter:
                        del self.suspicion_counter[cid]
                    filtered_results.append((client_proxy, fit_res))
            else:
                if cid not in self.blacklist:
                    filtered_results.append((client_proxy, fit_res))

        if not filtered_results:
            log(WARN, f"Nessun client valido al round {server_round}.")
            return None, {}

        if self.inplace:
            aggregated_ndarrays = aggregate_inplace(filtered_results)
        else:
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in filtered_results
            ]
            aggregated_ndarrays = aggregate(weights_results)


        self.previous_parameters = aggregated_ndarrays
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in filtered_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated
    
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

