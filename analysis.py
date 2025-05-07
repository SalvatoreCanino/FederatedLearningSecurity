import os
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from glob import glob
from garden.varG import get_case
import pandas as pd
from train_meta_classifier import MetaClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from train_meta_classifier import extract_weight_statistics
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
rf_model = joblib.load("./PIA/meta_model/rf_meta_model.pkl")

if get_case().endswith("_DIF"):
    LOG_DIR = "./property/property_logs_MD"
elif get_case().startswith("_PIA_DL"):
    LOG_DIR = "./property/property_logs_ddp"
elif get_case().startswith("_PIA_DQ"):
    LOG_DIR = "./property/property_logs_dq"
elif get_case().startswith("_PIA_DS"):
    LOG_DIR = "./property/property_logs_ds"
elif get_case().startswith("_PIA_DI"):
    LOG_DIR = "./property/property_logs_di"
else:
    LOG_DIR = "./property/property_logs"

META_MODEL_PATH = "./PIA/meta_model/meta_classifier_with_pca.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def softmax_with_temperature(logits, T=1.5):
    logits = logits / T
    exps = np.exp(logits - np.max(logits))
    return exps / exps.sum()

def load_meta_model(input_size):
    model = MetaClassifier(input_size=input_size).to(DEVICE)
    model.load_state_dict(torch.load(META_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def predict_distribution(meta_model, feature_vector):
    with torch.no_grad():
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        prediction = meta_model(feature_tensor).squeeze().cpu().numpy()
        distribution = softmax_with_temperature(prediction, T=1.5)
    return distribution


def analyze_class_mapping(csv_path, label_column='Label'):
  
    try:
        data = pd.read_csv(csv_path, sep=';', decimal=',')
        
        if label_column not in data.columns:
            available_cols = data.columns.tolist()
            print(f"\n❌ Colonna '{label_column}' non trovata!")
            print("Colonne disponibili:", available_cols)
            return None, None, None
        
        labels = data[label_column]
        unique_labels = labels.unique()
        
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        
        class_dist = pd.DataFrame({
            'Classe': le.classes_,
            'Codice': range(len(le.classes_)),
            'Conteggio': [sum(labels == cls) for cls in le.classes_],
            'Percentuale': [f"{(sum(labels == cls)/len(labels))*100:.1f}%" 
                          for cls in le.classes_]
        })
        
        print("Mapping delle classi:")
        print(class_dist.to_string(index=False))
        
        return class_mapping, le.classes_, data.head(3)
    
    except Exception as e:
        print(f"\nErrore durante l'analisi: {str(e)}")
        return None, None, None

def predict_distribution_rf(rf_model, feature_vector):
    pred = rf_model.predict([feature_vector])[0]
    pred = np.clip(pred, 0, 1)
    pred = pred / pred.sum()  
    return pred

if __name__ == "__main__":
    print(f"LOG_DIR selezionata: {LOG_DIR}")
        
    OUTPUT_PATH = f"./PIA/pd/predicted_distributions{get_case()}.json"

    weight_files = glob(os.path.join(LOG_DIR, f"round100_client*.npy"))

    if len(weight_files) == 0:
        print("Nessun file trovato. aaa")
        exit(1)

    flattened_weights_list = []
    client_ids = []

    for weight_path in weight_files:
        filename = os.path.basename(weight_path)
        client_id = filename.split("_client")[1].split(".")[0]
        client_ids.append(client_id)

        weights = np.load(weight_path)
        flattened_weights_list.append(weights)

    flattened_weights_array = np.vstack(flattened_weights_list)
    print(f"Dimensione pesi caricati: {flattened_weights_array.shape}")

    stat_features = [extract_weight_statistics(w) for w in flattened_weights_list]
    reduced_features = np.vstack(stat_features)

    input_size = reduced_features.shape[1]
    meta_model = load_meta_model(input_size)

    predictions = {}

    for i, client_id in enumerate(client_ids):
        feature_vector = reduced_features[i]
        pred_distribution = predict_distribution_rf(rf_model, feature_vector)

        predictions[client_id] = {
            "Acceptable": round(float(pred_distribution[0]), 4),
            "Hostile": round(float(pred_distribution[1]), 4),
            "Optimal": round(float(pred_distribution[2]), 4)
        }
            
    with open(OUTPUT_PATH, "w") as f:
        json.dump(predictions, f, indent=2)
        
    df = pd.DataFrame.from_dict(predictions, orient='index')
    df.index.name = "ClientID"
    df.to_csv(OUTPUT_PATH.replace(".json", ".csv"))

    print(f"\n{OUTPUT_PATH}")