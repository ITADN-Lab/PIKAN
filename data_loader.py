# 数据加载模块
import sqlite3
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def fetch_balanced_samples(db_path, num_samples_per_label):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT label FROM samples")
    labels = [row[0] for row in cursor.fetchall()]

    balanced_samples = []

    for label in labels:
        cursor.execute("SELECT * FROM samples WHERE label = ?", (label,))
        rows = cursor.fetchall()

        if len(rows) < num_samples_per_label:
            selected_rows = rows
        else:
            selected_rows = random.sample(rows, num_samples_per_label)

        for row in selected_rows:
            features = list(row[2:])
            balanced_samples.append((features, row[1]))

    conn.close()
    return balanced_samples


class FlowDataset(Dataset):
    def __init__(self, samples):
        self.X = torch.tensor([s[0][:50] for s in samples], dtype=torch.float32)
        self.y_voltage = torch.tensor([s[0][50:110] for s in samples], dtype=torch.float32)

        self.mean_X = np.nanmean(self.X, axis=0)
        self.std_X = np.nanstd(self.X, axis=0) + 1e-8
        self.X = (self.X - self.mean_X) / self.std_X

        self.vm_mean = np.nanmean(self.y_voltage[:, ::2], axis=0)
        self.vm_std = np.nanstd(self.y_voltage[:, ::2], axis=0) + 1e-8

        self.va_mean = np.nanmean(self.y_voltage[:, 1::2], axis=0)
        self.va_std = np.nanstd(self.y_voltage[:, 1::2], axis=0) + 1e-8

        self.y = self.y_voltage.clone()
        self.y[:, ::2] = (self.y_voltage[:, ::2] - self.vm_mean) / self.vm_std
        self.y[:, 1::2] = (self.y_voltage[:, 1::2] - self.va_mean) / self.va_std

        self.vm_mean = torch.tensor(self.vm_mean, dtype=torch.float32)
        self.vm_std = torch.tensor(self.vm_std, dtype=torch.float32)
        self.va_mean = torch.tensor(self.va_mean, dtype=torch.float32)
        self.va_std = torch.tensor(self.va_std, dtype=torch.float32)

    def denormalize_X(self, x_tensor):
        mean_X_tensor = torch.tensor(self.mean_X, dtype=torch.float32).to(x_tensor.device)
        std_X_tensor = torch.tensor(self.std_X, dtype=torch.float32).to(x_tensor.device)
        return x_tensor * std_X_tensor + mean_X_tensor

    def __len__(self):
        return len(self.y_voltage)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(db_path, num_samples_per_label, dataset_split_rate, batch_size):
    samples = fetch_balanced_samples(db_path, num_samples_per_label)
    dataset = FlowDataset(samples)

    train_size = int(dataset_split_rate * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, train_loader, test_loader