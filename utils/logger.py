import csv, os, numpy as np


class Logger:
    def __init__(self, path, fields):
        self.path = path; self.fields = fields
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,'w',newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def write(self, row):
        with open(self.path,'a',newline='') as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)

    @staticmethod
    def save_history(history, path):
        np.save(path, history)
        print(f"[Logger] History saved → {path}")

    @staticmethod
    def load_history(path):
        return np.load(path, allow_pickle=True).item()
