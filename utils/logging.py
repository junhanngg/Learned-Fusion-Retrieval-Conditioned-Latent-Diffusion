# import packages
import os
import csv

"""
Save history dictionary to csv
"""
def save_history(history, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    keys = list(history.keys())
    num_rows = len(history[keys[0]])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(num_rows):
            writer.writerow([history[k][i] for k in keys])

    print(f"Saved history CSV to: {csv_path}")