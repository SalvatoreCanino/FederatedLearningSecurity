"""garden: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import pandas as pd
from flwr.server.strategy import FedAvg
from sklearn.utils.class_weight import compute_class_weight
from flwr.common import Context
from sklearn.metrics import f1_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import csv
import random
from garden.varG import get_case

global pid
pid= os.getpid()

first_access = [True, True, True, True]
class_weights = [None, None, None, None]
class_mapping = [None, None, None, None]

dati = ""
case = get_case()

class LSTMNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=512, output_size=3, num_layers=2):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)              
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class GRUNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=512, output_size=3, num_layers=2):
        super(GRUNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def load_data(partition_id: int):

    data = pd.read_csv(f"../dati/Dati_classificati{dati}{partition_id+1}.csv", sep=';')  
    
    X = data[['Um', 'Temp']].replace(',', '.', regex=True).astype(float).values  
    y = data['Label']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    class_mapping[pid%4] = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    class_weights[pid%4] = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights[pid%4] = torch.tensor(class_weights[pid%4], dtype=torch.float32)

    if case.endswith("_DIF"):    
        # --- Data Sanitization (Label Noise Detection con k-NN) ---
        n_neighbors_check = 9
        initial_rows = len(y)

        if initial_rows > n_neighbors_check: 
            temp_scaler = StandardScaler()
            X_scaled_temp = temp_scaler.fit_transform(X)

            knn = KNeighborsClassifier(n_neighbors=n_neighbors_check)
            knn.fit(X_scaled_temp, y)
            y_pred_knn = knn.predict(X_scaled_temp)

            suspicious_indices = np.where(y != y_pred_knn)[0]
            num_suspicious = len(suspicious_indices)

            if num_suspicious > 0:
                keep_mask = np.ones(initial_rows, dtype=bool)
                keep_mask[suspicious_indices] = False

                X = X[keep_mask]
                y = y[keep_mask] 

    if len(y) == 0:
         empty_tensor = torch.empty((0, X.shape[1]), dtype=torch.float32)
         empty_labels = torch.empty((0,), dtype=torch.long)
         train_dataset = torch.utils.data.TensorDataset(empty_tensor, empty_labels)
         test_dataset = torch.utils.data.TensorDataset(empty_tensor, empty_labels)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

    return (
        DataLoader(train_dataset, batch_size=1024, shuffle=True),
        DataLoader(test_dataset, batch_size=1024, shuffle=False)
    )

# Load with Attack DPA
def load_data_DPA(partition_id: int, attack_type="untargeted", attack_prob=0.3):
    
    data = pd.read_csv(f"../dati/Dati_classificati{dati}{partition_id+1}.csv", sep=';')
    X = data[['Um', 'Temp']].replace(',', '.', regex=True).astype(float).values
    y = data['Label']
    
    if partition_id+1 == 1:  
        if attack_type == "untargeted":
            mask = np.random.rand(len(y)) < attack_prob
            y[mask] = y[mask].apply(lambda x: random.choice(["Hostile", "Acceptable", "Optimal"]))
            
        elif attack_type == "targeted":
            trigger_temp, trigger_um = 25, 55
            trigger_label = "Hostile"

            for i in range(len(X)):
                if (X[i][0] - trigger_temp)**2 + (X[i][1] - trigger_um)**2 <= 25:
                    y[i] = trigger_label 

            num_trigger_points = int(len(X) * 0.1)  

            X_trigger = np.array([[trigger_um, trigger_temp]] * num_trigger_points)
            y_trigger = np.array([trigger_label] * num_trigger_points)

            X = np.concatenate([X, X_trigger])
            y = np.concatenate([y, y_trigger])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    class_mapping[pid%4] = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    class_weights[pid%4] = torch.tensor(
        compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y),
        dtype=torch.float32
    )
 
    if (case.startswith("_DPA_U_D") or case.endswith("_DIF")) and (partition_id+1 == 1):    
        # --- Data Sanitization (Label Noise Detection con k-NN) ---
        n_neighbors_check = 9
        initial_rows = len(y)

        if initial_rows > n_neighbors_check: 
            temp_scaler = StandardScaler()
            X_scaled_temp = temp_scaler.fit_transform(X)

            knn = KNeighborsClassifier(n_neighbors=n_neighbors_check)
            knn.fit(X_scaled_temp, y)
            y_pred_knn = knn.predict(X_scaled_temp)

            suspicious_indices = np.where(y != y_pred_knn)[0]
            num_suspicious = len(suspicious_indices)

            if num_suspicious > 0:
                keep_mask = np.ones(initial_rows, dtype=bool)
                keep_mask[suspicious_indices] = False

                X = X[keep_mask]
                y = y[keep_mask] 

    if len(y) == 0:
         empty_tensor = torch.empty((0, X.shape[1]), dtype=torch.float32)
         empty_labels = torch.empty((0,), dtype=torch.long)
         train_dataset = torch.utils.data.TensorDataset(empty_tensor, empty_labels)
         test_dataset = torch.utils.data.TensorDataset(empty_tensor, empty_labels)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

    return (
        DataLoader(train_dataset, batch_size=1024, shuffle=True),
        DataLoader(test_dataset, batch_size=1024, shuffle=False)
    )


def train(net, trainloader, epochs, lr_server, device):
    net.to(device)  
    weights = class_weights[pid%4]
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(net.parameters(), lr = lr_server)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=10)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        running_loss = 0.0
        for umtemp, labels in trainloader:  
            optimizer.zero_grad()
            outputs = net(umtemp.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    avg_trainloss = running_loss / len(trainloader)
    scheduler.step(avg_trainloss)
    return avg_trainloss


def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for umtemp, labels in testloader:
            outputs = net(umtemp.to(device))
            loss += criterion(outputs, labels.to(device)).item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels.cpu().numpy()).sum()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    current_class_mapping = class_mapping[pid % 4]
    
    inv_class_mapping = {v: k for k, v in current_class_mapping.items()}
    
    class_labels = sorted(current_class_mapping.values())
    
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=class_labels)
    
    f1_scores = {
        inv_class_mapping[class_labels[i]]: f1_per_class[i] 
        for i in range(len(class_labels))
    }
    
    f1_acceptable = f1_scores.get('Acceptable', 0.0)
    f1_hostile = f1_scores.get('Hostile', 0.0)
    f1_optimal = f1_scores.get('Optimal', 0.0)

    save_metrics_to_csv(f"../risultati/grezzi/metrics{case}{pid % 4 + 1}.csv", loss, accuracy, 
                       f1_weighted, f1_acceptable, f1_hostile, f1_optimal)
    
    return loss, accuracy, f1_weighted , f1_acceptable, f1_hostile, f1_optimal

def save_metrics_to_csv(file_name, loss_values, accuracy_values, f1_weighted, f1_acceptable, f1_hostile, f1_optimal) -> None:
    
    headers = ["loss", "accuracy", "f1_weighted", "f1_acceptable", "f1_hostile", "f1_optimal"]
    values = [ loss_values, accuracy_values, f1_weighted, f1_acceptable, f1_hostile, f1_optimal,]

    file_exists = os.path.isfile(file_name)
    mode = "w" if first_access[pid % 4] else "a"

    with open(file_name, mode=mode, newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        if first_access[pid % 4]:
            writer.writerow(headers)
            first_access[pid % 4] = False  
        
        writer.writerow(values)

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
