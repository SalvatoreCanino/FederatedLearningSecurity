import csv
import openpyxl
from statistics import mean
from openpyxl.styles import Alignment
from garden.varG import get_case
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
from plot import png

def auto_adjust_column_width(ws):
    """Regola automaticamente la larghezza delle colonne in base al contenuto della prima riga e gestisce le colonne vuote"""
    for col in ws.columns:
        column = col[0].column_letter  # Prende la lettera della colonna
        first_cell_value = str(col[0].value)  # Controlla solo la prima cella per determinare la larghezza
        
        if first_cell_value:  # Se la prima cella non è vuota
            max_length = len(first_cell_value)
            ws.column_dimensions[column].width = max_length * 1.2  # Adatta la larghezza in base al contenuto
        else:  # Se la prima cella è vuota, imposta la larghezza predefinita
            ws.column_dimensions[column].width = 8.43  # Larghezza predefinita di Excel

def read_csv(file_name):
    """Legge il file CSV e restituisce i dati come lista di dizionari"""
    with open(file_name, mode='r') as csv_file:
        return list(csv.DictReader(csv_file))

def write_to_excel(output_file, data_list):
    """Scrive i dati nel file Excel con la struttura specificata"""
    wb = openpyxl.Workbook()
    ws = wb.active
    
    # Creazione header
    headers = []
    for i in range(1, 5):
        headers.extend([
            f"Loss{i}", f"Accuracy{i}", f"F1Weighted{i}",
            f"F1Acceptable{i}", f"F1Hostile{i}", f"F1Optimal{i}"
        ])
        headers.append("")  # Separatore

    headers.extend([
        "Avg_Loss", "Avg_Accuracy", "Avg_F1Weighted",
        "Avg_F1Acceptable", "Avg_F1Hostile", "Avg_F1Optimal"
    ])
    
    ws.append(headers)
    
    # Scrittura dati
    max_rows = max(len(data) for data in data_list)
    for row_idx in range(max_rows):
        row_data = []
        metrics = {
            'loss': [], 'accuracy': [], 'f1_weighted': [],
            'f1_acceptable': [], 'f1_hostile': [], 'f1_optimal': []
        }
        
        for client_data in data_list:
            if row_idx < len(client_data):
                data = client_data[row_idx]
                row_data.extend([
                    float(data['loss']),
                    float(data['accuracy']),
                    float(data['f1_weighted']),
                    float(data['f1_acceptable']),
                    float(data['f1_hostile']),
                    float(data['f1_optimal'])
                ])
                # Aggiungi ai valori per le medie
                for key in metrics:
                    metrics[key].append(float(data[key]))
            else:
                row_data.extend([""] * 6)
            
            row_data.append("")  # Separatore dopo ogni client
        
        # Calcola medie
        row_data.extend([
            mean(metrics['loss']) if metrics['loss'] else "",
            mean(metrics['accuracy']) if metrics['accuracy'] else "",
            mean(metrics['f1_weighted']) if metrics['f1_weighted'] else "",
            mean(metrics['f1_acceptable']) if metrics['f1_acceptable'] else "",
            mean(metrics['f1_hostile']) if metrics['f1_hostile'] else "",
            mean(metrics['f1_optimal']) if metrics['f1_optimal'] else "",
        ])
        
        ws.append(row_data)
    
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center")

    for col in ws.columns:
        for cell in col:
            if isinstance(cell.value, (int, float)):
                cell.number_format = '0.000'

    auto_adjust_column_width(ws)
    
    wb.save(output_file)

if __name__ == "__main__":
    
    case = get_case()
    csv_files = [f"../risultati/grezzi/metrics{case}{i + 1}.csv" for i in range(4)]
    data_list = [read_csv(f) for f in csv_files]
    write_to_excel(f"../risultati/metrics_final{case}.xlsx", data_list)
    png(case)
