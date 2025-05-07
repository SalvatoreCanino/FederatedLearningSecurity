import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
##################secondo

SHADOW_DIR = "./PIA/shadow_models_smart"  
META_MODEL_DIR = "./PIA/meta_model"
BATCH_SIZE = 16
EPOCHS = 300
TARGET_PCA_DIM = 95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(META_MODEL_DIR, exist_ok=True)

class MetaDataset(Dataset):
    def __init__(self, feature_vectors, label_distributions):
        self.features = feature_vectors
        self.labels = label_distributions

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

class MetaClassifier(nn.Module):
    def __init__(self, input_size, output_size=3):
        super(MetaClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)

def extract_weight_statistics(flat_weights):
    hist, _ = np.histogram(flat_weights, bins=50, density=True)
    hist = hist[hist > 0]

    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(flat_weights.reshape(-1, 1))
    gmm_means = gmm.means_.flatten()
    gmm_weights = gmm.weights_.flatten()

    stats_dict = {
        "mean": np.mean(flat_weights),
        "std": np.std(flat_weights),
        "skewness": stats.skew(flat_weights),
        "kurtosis": stats.kurtosis(flat_weights),
        "entropy": -np.sum(hist * np.log(hist + 1e-10)),
        "l2_norm": np.linalg.norm(flat_weights, ord=2),
        "max_norm": np.max(np.abs(flat_weights)),
        "quantile_25": np.percentile(flat_weights, 25),
        "quantile_50": np.percentile(flat_weights, 50),
        "quantile_75": np.percentile(flat_weights, 75),
        "ptp": np.ptp(flat_weights),  # max - min
        "median": np.median(flat_weights),
        "iqr": np.percentile(flat_weights, 75) - np.percentile(flat_weights, 25),
        "mad": np.median(np.abs(flat_weights - np.median(flat_weights))),
        "coef_var": np.std(flat_weights) / (np.mean(flat_weights) + 1e-8),
        "zero_crossings": ((flat_weights[:-1] * flat_weights[1:]) < 0).sum(),
        "sparsity": np.mean(np.abs(flat_weights) < 1e-3)
    }

    for i in range(3):
        stats_dict[f"gmm_mean_{i}"] = gmm_means[i]
        stats_dict[f"gmm_weight_{i}"] = gmm_weights[i]

    return np.array(list(stats_dict.values()))

def load_features(shadow_dir):
    feature_vectors = []
    label_distributions = []

    with open(os.path.join(shadow_dir, "label_distributions.json"), "r") as f:
        label_info = json.load(f)

    for key, distribution in label_info.items():
        shadow_id = int(key.replace("shadow", ""))
        weights_path = os.path.join(shadow_dir, f"weights_shadow{shadow_id}.npy")

        if not os.path.exists(weights_path):
            continue

        weights = np.load(weights_path)
        stats_vector = extract_weight_statistics(weights)  
        feature_vectors.append(stats_vector)

        label_distributions.append([
            distribution["Acceptable"],
            distribution["Hostile"],
            distribution["Optimal"]
        ])

    return np.array(feature_vectors), np.array(label_distributions)

def apply_and_save_pca(features, target_dim, save_path):
    print(f"Applicando PCA: {features.shape} --> {target_dim} dimensioni...")
    pca = PCA(n_components=target_dim)
    reduced_features = pca.fit_transform(features)
    joblib.dump(pca, save_path)
    print(f"✅ PCA salvata su {save_path}")
    return reduced_features

def smooth_labels(y, epsilon=0.05):
    return y * (1 - epsilon) + epsilon / y.shape[1]

def train_meta_classifier(X_train, y_train, X_val, y_val, input_size):
    train_dataset = MetaDataset(X_train, y_train)
    val_dataset = MetaDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MetaClassifier(input_size=input_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)        
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            labels = smooth_labels(labels, epsilon=0.05)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} - Loss: {loss / len(train_loader):.4f}")

    model.eval()
    y_preds = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).cpu().numpy()
            y_preds.append(outputs)
    y_preds = np.concatenate(y_preds, axis=0)

    return model, y_preds

if __name__ == "__main__":
    print("Caricando feature vectors...")
    feature_vectors, label_distributions = load_features(SHADOW_DIR)

    print(f"Feature vectors caricati: {feature_vectors.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        feature_vectors,
        label_distributions,
        test_size=0.2,
        random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_preds = rf.predict(X_val)

    mse = mean_squared_error(y_val, y_preds)
    cos_sim = np.mean([
        cosine_similarity([true], [pred])[0, 0]
        for true, pred in zip(y_val, y_preds)
    ])
    true_majority = np.argmax(y_val, axis=1)
    pred_majority = np.argmax(y_preds, axis=1)
    majority_acc = np.mean(true_majority == pred_majority)

    print(f"MSE: {mse:.6f}")
    print(f"Similarità Coseno media: {cos_sim:.4f}")
    print(f"Accuracy classe dominante: {majority_acc:.4f}")

    joblib.dump(rf, os.path.join(META_MODEL_DIR, "rf_meta_model.pkl"))
