# FederatedLearningSecurity
## Project Overview

This project implements a federated learning system for monitoring and optimizing microclimatic parameters in agricultural greenhouses using data from four distinct greenhouses. The primary objective is to develop and evaluate defense strategies to protect federated models from potential interference, ensuring the robustness and integrity of the system.

The project leverages the **Flower (FL)** framework to manage the federated learning process, integrating various defense strategies and optimization techniques, including:

- Quantization and Sparsification  
- Clipping and Masking  
- Detection of Malicious Clients (Free Riders)  
- Trust Score-Based Aggregation  
- Differential Privacy-Based Strategies  

---

## Global Structure 
```
Project:.
+---dati
+---ENVIRONMENT
+---garden
¦   +---garden
¦   ¦   +---client_app.py
¦   ¦   +---free_rider_detection_strategy.py
¦   ¦   +---mega_defence_strategy.py
¦   ¦   +---property_inference_strategy.py
¦   ¦   +---server_app.py
¦   ¦   +---task.py
¦   ¦   +---trust_strategy.py
¦   ¦   +---varG.py
¦   +---analysis.py
¦   +---final.py
¦   +---generate_shadow_models.py
¦   +---plot.py
¦   +---train_meta_classifier.py
+---risultati
¦   +---grezzi

```

---

## Project Structure

- **client_app.py:** Implementation of the federated client, local training management, and integration of defense strategies.  
- **server_app.py:** Management of the federated server, strategy selection, and configuration of global parameters.  
- **task.py:** Definition of neural network models (LSTM, GRU) and handling of local datasets for each greenhouse.  
- **varG.py:** Configuration of global parameters, use case selection, and number of training rounds.  
- **free_rider_detection_strategy.py:** Implementation of the Free Rider detection strategy based on dynamic thresholds and weight updates analysis.  
- **mega_defence_strategy.py:** Advanced defense strategy that combines anomaly detection, clipping, and trust score management.  
- **property_inference_strategy.py:** Implementation of strategies for protecting sensitive data using masking and quantization techniques.  
- **trust_strategy.py:** Trust score-based aggregation system to penalize suspicious clients and prioritize reliable ones.  

---

## Dependencies

- Python 3.9+  
- Flower 
---

## Execution Instructions

1. Clone the repository:  
   ```
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required dependencies:  
   
   https://www.youtube.com/watch?v=cRebUIGB5RU&list=PLNG4feLHqCWlnj8a_E1A_n5zr2-8pafTB&ab_channel=Flower
   Follow this guide; I strongly recommend watching the entire playlist.

4. Run the project
   ```
   flwr run .
   ```
---

## Parameter Configuration

The configuration of the use case (type of attack/defense strategy) is defined in the `varG.py` file.  
Possible values for the `case` parameter include:

- `_DPA_U`: Untargeted Data Poisoning  
- `_PIA`: Property Inference Attack  
- `_FRA`: Free Rider Attack  
- `_MUPA_BA`: Model Update Poisoning Attack  
- `_DIF`: Final Defense (advanced strategy)  

---

## Other Scripts

- **final.py**: Aggregates evaluation metrics from multiple clients and compiles them into a single Excel file, including calculated averages for loss, accuracy, and F1 scores.

- **plot.py**: Generates plots of average metrics (loss, accuracy, F1 score) across server rounds and comparison bar plots for defense strategies.

- **generate_shadow_models.py**: Creates synthetic shadow datasets and trains shadow models to simulate various label distributions and client behaviors.

- **train_meta_classifier.py**: Trains a meta-classifier to predict label distributions based on model weights extracted from shadow models.

- **analysis.py**: Applies the meta-classifier to predict label distributions for each client based on the extracted weight statistics and saves the results in JSON and CSV formats.

---

## Author  

- **Salvatore Canino**  
- Email: salvatorecainino1@gmail.com  
- University of Pisa  
- Year: 2025  

