import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from itertools import product
from garden.task import GRUNetwork
import numpy as np

OUTPUT_DATASET_DIR = "./PIA/shadow_datasets_smart"
OUTPUT_MODEL_DIR = "./PIA/shadow_models_smart"
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

DATASET_SIZE = 30000
BATCH_SIZE = 1024
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_dataset(size, acceptable_ratio, hostile_ratio, optimal_ratio):
    n_acceptable = int(size * acceptable_ratio)
    n_hostile = int(size * hostile_ratio)
    n_optimal = size - n_acceptable - n_hostile

    temps, hums, labels = [], [], []

    # Acceptable
    for _ in range(n_acceptable):
        choice = np.random.choice(["temp_medium", "hum_medium"])
        if choice == "temp_medium":
            temp = np.random.uniform(18, 29)
            hum = np.random.choice([np.random.uniform(0, 39), np.random.uniform(66, 100)])
        else:
            hum = np.random.uniform(40, 65)
            temp = np.random.choice([np.random.uniform(0, 17), np.random.uniform(30, 45)])
        temps.append(temp)
        hums.append(hum)
        labels.append("Acceptable")

    # Hostile
    for _ in range(n_hostile):
        temp = np.random.choice([np.random.uniform(0, 17), np.random.uniform(30, 45)])
        hum = np.random.choice([np.random.uniform(0, 39), np.random.uniform(66, 100)])
        temps.append(temp)
        hums.append(hum)
        labels.append("Hostile")

    # Optimal
    for _ in range(n_optimal):
        temp = np.random.uniform(20, 26)
        hum = np.random.uniform(45, 55)
        temps.append(temp)
        hums.append(hum)
        labels.append("Optimal")

    # Shuffle
    temps = np.array(temps)
    hums = np.array(hums)
    labels = np.array(labels)
    indices = np.arange(size)
    np.random.shuffle(indices)

    dataset = pd.DataFrame({
        "Um": hums[indices],
        "Temp": temps[indices],
        "Label": labels[indices]
    })
    return dataset

class ShadowDataset(torch.utils.data.Dataset):
    def __init__(self, temps, hums, labels, label_to_idx):
        self.X = np.stack([temps, hums], axis=1)
        self.y = np.array([label_to_idx[label] for label in labels])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def train_shadow_model(temps, hums, labels):
    label_to_idx = {"Acceptable": 0, "Hostile": 1, "Optimal": 2}
    dataset = ShadowDataset(temps, hums, labels, label_to_idx)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GRUNetwork(input_size=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    label_distributions = {}
    current_id = 0
    steps = np.round(np.arange(0.05, 0.91, 0.05), 2)  
    combinations = [
        (a, h, o) for a, h, o in product(steps, repeat=3)
        if np.isclose(a + h + o, 1.0, atol=0.001)
    ]

    print(f"Trovate {len(combinations)} combinazioni valide con somma 1.0")

    current_id = 0
    label_distributions = {}

    for acceptable_ratio, hostile_ratio, optimal_ratio in tqdm(combinations):

        dataset = generate_dataset(DATASET_SIZE, acceptable_ratio, hostile_ratio, optimal_ratio)
        output_path = os.path.join(OUTPUT_DATASET_DIR, f"shadow_dataset_{current_id}.csv")
        dataset.to_csv(output_path, index=False)

        temps = dataset["Temp"].values
        hums = dataset["Um"].values
        labels = dataset["Label"].values

        label_distributions[f"shadow{current_id}"] = {
            "Acceptable": round(acceptable_ratio, 3),
            "Hostile": round(hostile_ratio, 3),
            "Optimal": round(optimal_ratio, 3),
        }

        model = train_shadow_model(temps, hums, labels)
        model_params = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
        flat_weights = np.concatenate(model_params)
        np.save(os.path.join(OUTPUT_MODEL_DIR, f"weights_shadow{current_id}.npy"), flat_weights)

        current_id += 1

    with open(os.path.join(OUTPUT_MODEL_DIR, "label_distributions.json"), "w") as f:
        json.dump(label_distributions, f, indent=2)

    print(f"\nCompletato: {current_id} shadow models generati e salvati.")