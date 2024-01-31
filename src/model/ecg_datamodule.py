import os
from typing import Optional
from pathlib import Path
import numpy as np

import wfdb
from wfdb.processing import resample_sig
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import numpy.typing as npt
from model.ecg_dataset import ECGDataset
from common.utils import parse_annotations
from common.data_classes import Range, AnnotationSymbols, leads

RANDOM_STATE_SEED = 42


class ECGDataModule(LightningDataModule):
    def __init__(
        self,
        ecg_dir: Path,
        batch_size: int = 4,
        num_of_workers: int | None = os.cpu_count(),
        train_ratio: float = 0.8,
        resample_fs: int = 100
    ) -> None:
        super().__init__()
        self.ecg_dir = ecg_dir
        self.batch_size = batch_size
        self.num_of_workers = num_of_workers if num_of_workers else 0
        self.train_ratio = train_ratio
        self.resample_fs = resample_fs

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: Optional[str] = None) -> None:
        with (self.ecg_dir / "RECORDS").open() as f:
            record_names = f.read().splitlines()

        # Load records and annotations
        records = []
        annotations = []
        for record_name in record_names:
            record_path = self.ecg_dir / record_name
            record = wfdb.rdrecord(record_path)
            p_signals = np.array([resample_sig(record.p_signal[:, i], record.fs, self.resample_fs)[0] for i in range(record.p_signal.shape[1])])
            lead_annotations = []
            for lead in leads:
                annotation = wfdb.rdann(str(record_path), lead)
                parsed_annotation = parse_annotations(annotation, self.resample_fs)
                lead_annotations.append(self._generate_result(parsed_annotation, p_signals.shape[1]))

            records.append(np.asarray(p_signals, dtype=np.float32))
            annotations.append(lead_annotations)

        # Split the dataset into training, validation, and test sets
        train_records, test_records = train_test_split(
            list(zip(records, annotations)), train_size=self.train_ratio, random_state=RANDOM_STATE_SEED
        )
        val_records, test_records = train_test_split(test_records, train_size=0.5, random_state=RANDOM_STATE_SEED)

        # Unzip the records and annotations
        train_records, train_annotations = zip(*train_records)
        val_records, val_annotations = zip(*val_records)
        test_records, test_annotations = zip(*test_records)

        # Create datasets
        self.train = ECGDataset(np.asarray(train_records), np.asarray(train_annotations))
        self.val = ECGDataset(np.asarray(val_records), np.asarray(val_annotations))
        self.test = ECGDataset(np.asarray(test_records), np.asarray(test_annotations))


    def _generate_result(self, symbols: AnnotationSymbols, length: int):
        def generate_signal(ranges: list[Range], result: npt.NDArray, class_num: int) -> np.ndarray:
            for interval in ranges:
                result[interval.start : interval.end] = class_num
            return result

        sig = np.zeros(length,dtype=np.int64)
        sig = generate_signal(symbols.p, sig, 1)
        sig = generate_signal(symbols.N, sig, 2)
        sig = generate_signal(symbols.t, sig, 3)
        return sig

    def train_dataloader(self) -> DataLoader:
        if not self.train:
            raise Exception("need to call setup() first")
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_of_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if not self.val:
            raise Exception("need to call setup() first")
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_of_workers)

    def test_dataloader(self) -> DataLoader:
        if not self.test:
            raise Exception("need to call setup() first")
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_of_workers)
