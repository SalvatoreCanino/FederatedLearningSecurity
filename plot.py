import csv
import openpyxl
from statistics import mean
from openpyxl.styles import Alignment
from garden.varG import get_case
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def png(case):

    file_pattern = f"../risultati/metrics_final{case[:4]}*.xlsx"

    if case.startswith("_MUPA"):
        basis_file_pattern = f"../risultati/metrics_final100.xlsx"
    elif case.startswith("_FRA"):
        basis_file_pattern = f"../risultati/metrics_final40.xlsx"
    else:
        basis_file_pattern = f"../risultati/metrics_final.xlsx"
    
    if case.endswith("_DIF") or case.endswith("100"):
        basis_file_pattern = f"../risultati/metrics_final100.xlsx"
        file_pattern = f"../risultati/metrics_final*_DIF.xlsx"
    basis_file_list = glob.glob(basis_file_pattern)
    files = glob.glob(file_pattern)
    basis_file_list += files    

    if case.endswith("100"):
        attack_file_pattern = f"../risultati/metrics_final_*100.xlsx"
        attack_file_list = glob.glob(attack_file_pattern)
        basis_file_list += attack_file_list
    
    if not basis_file_list:
        print("Nessun file trovato con il pattern specificato.")
    else:
        avg_loss_data = {}
        avg_accuracy_data = {}
        avg_f1_weighted_data = {}
        
        for file in basis_file_list:
            try:
                df = pd.read_excel(file)
                name = os.path.basename(file)
                avg_loss_data[name] = df['Avg_Loss']
                avg_accuracy_data[name] = df['Avg_Accuracy']
                avg_f1_weighted_data[name] = df['Avg_F1Weighted']
            except Exception as e:
                print(f"Errore con il file {file}: {e}")

        plt.figure(figsize=(10, 6))
        for name, data in avg_loss_data.items():
            plt.plot(data, label=name)
        plt.title("Avg_Loss per file")
        plt.xlabel("Server Round")
        plt.ylabel("Avg_Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_path = "../risultati/avg_loss_plot.png"
        if os.path.exists(loss_path):
            os.remove(loss_path)
        plt.savefig(loss_path)

        plt.figure(figsize=(10, 6))
        for name, data in avg_accuracy_data.items():
            plt.plot(data, label=name)
        plt.title("Avg_Accuracy per file")
        plt.xlabel("Server Round")
        plt.ylabel("Avg_Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        acc_path = "../risultati/avg_accuracy_plot.png"
        if os.path.exists(acc_path):
            os.remove(acc_path)
        plt.savefig(acc_path)

        plt.figure(figsize=(10, 6))
        for name, data in avg_f1_weighted_data.items():
            plt.plot(data, label=name)
        plt.title("Avg_F1Weighted per file")
        plt.xlabel("Server Round")
        plt.ylabel("Avg_F1Weighted")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        f1_path = "../risultati/avg_f1_weighted_plot.png"
        if os.path.exists(f1_path):
            os.remove(f1_path)
        plt.savefig(f1_path)

def bar_plot(case):
    
    base_path = f"../risultati/metrics_final{case}_DIF.xlsx"
    dif_base_pattern = f"../risultati/metrics_final{case}*.xlsx"
    dif_base_list = [f for f in glob.glob(dif_base_pattern) if os.path.abspath(f) != os.path.abspath(base_path)]

    if not dif_base_list:
        print("Nessun file trovato con il pattern specificato.")
        return

    max_value = -1
    best_file = None
    for file in dif_base_list:
        df = pd.read_excel(file)
        if 'Avg_F1Weighted' in df.columns:
            last_value = df['Avg_F1Weighted'].dropna().iloc[-1]
            if last_value > max_value:
                max_value = last_value
                best_file = file

    if best_file is None:
        print("Nessun file valido trovato con la colonna 'Avg_F1Weighted'.")
        return

    base_df = pd.read_excel(base_path)
    if 'Avg_F1Weighted' not in base_df.columns:
        print("Il file base non contiene la colonna 'Avg_F1Weighted'.")
        return

    base_value = base_df['Avg_F1Weighted'].dropna().iloc[-1]
    best_file = best_file.split("_")[-1]
    labels = ['Difesa Totale', best_file]
    values = [base_value, max_value]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    bars = plt.bar(labels, values)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom')

    plt.ylabel('Avg_F1Weighted finale')
    plt.title(f'Confronto performance finali - caso {case}')
    plt.ylim(0, 1) 
    plot_path = f"../risultati/comparison{case}.png"
    if os.path.exists(plot_path):
        os.remove(plot_path)
    plt.savefig(plot_path)


if __name__ == "__main__":
    
    case = get_case()
    png(case)
    bar_plot(case)
