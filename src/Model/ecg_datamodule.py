from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
import wfdb
import os
from sklearn.model_selection import train_test_split

from Model.ecg_dataset import ECGDataset

class ECGDataModule(LightningDataModule):
    def __init__(
        self, ecg_dir: str, batch_size: int = 4, num_of_workers: int = 8,
        train_ratio: float = 0.8, transform: Optional[transforms.Compose] = None
    ) -> None:
        super().__init__()
        self.ecg_dir = ecg_dir
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers
        self.train_ratio = train_ratio
        self.transform = transform

        self.leads = [
            'i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'
        ]

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: Optional[str] = None) -> None:
        with open(os.path.join(self.ecg_dir, 'RECORDS')) as f:
            record_names = f.read().splitlines()

        # Load records and annotations
        records = []
        annotations = []
        for record_name in record_names:
            record_path = os.path.join(self.ecg_dir, record_name)
            record = wfdb.rdrecord(record_path)
            lead_annotations = {}
            for lead in self.leads:
                annotation = wfdb.rdann(record_path, lead)
                parsed_annotation = self.parse_annotations(annotation)
                lead_annotations[lead] = parsed_annotation
            
            records.append(record)
            annotations.append(lead_annotations)

        # Split the dataset into training, validation, and test sets
        train_records, test_records = train_test_split(
            list(zip(records, annotations)), train_size=self.train_ratio
        )
        val_records, test_records = train_test_split(test_records, train_size=0.5)

        # Unzip the records and annotations
        train_records, train_annotations = zip(*train_records)
        val_records, val_annotations = zip(*val_records)
        test_records, test_annotations = zip(*test_records)

        # Create datasets
        self.train = ECGDataset(train_records, train_annotations, transform=self.transform)
        self.val = ECGDataset(val_records, val_annotations, transform=self.transform)
        self.test = ECGDataset(test_records, test_annotations, transform=self.transform)

    def parse_annotations(annotation):
        intervals = {'p': [], 'N': [], 't': []}

        for symbol, index in zip(annotation.symbol, annotation.sample):
            if symbol == '(':
                current_start = index
            elif symbol in ['p', 'N', 't']:
                current_symbol = symbol
                current_peak = index
            elif symbol == ')' and current_symbol:
                intervals[current_symbol].append([current_start, current_peak, index])

        return intervals

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_of_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_of_workers
        )