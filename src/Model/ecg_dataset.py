import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, records, annotations):
        self.records = records
        self.annotations = annotations

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        annotation = self.annotations[idx]

        return torch.tensor(record, dtype=torch.float32), annotation
