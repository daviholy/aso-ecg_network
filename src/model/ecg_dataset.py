import numpy.typing as npt
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, records: npt.NDArray, annotations: npt.NDArray) -> None:
        self.records = torch.as_tensor(records)
        self.annotations = torch.as_tensor(annotations)

    def __len__(self) -> int:
        return len(self.records) * 12

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx // 12, idx % 12]
        annotation = self.annotations[idx // 12, idx % 12]
        return record, annotation
