import os
from typing import Optional, cast
from pathlib import Path
import numpy as np

import wfdb
from wfdb import Annotation
from wfdb.processing import resample_sig, resample_ann
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import numpy.typing as npt
from model.ecg_dataset import ECGDataset
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
            for lead_idx, lead in enumerate(leads):
                annotation = wfdb.rdann(str(record_path), lead)
                parsed_annotation = self.parse_annotations(annotation, self.resample_fs)

                first_annot_idx = parsed_annotation.get_min_start_idx()
                last_annot_idx = parsed_annotation.get_max_end_idx()

                threshold_annot_shift = int(0.02 * len(p_signals[lead_idx]))

                start_cut_off = max(0, first_annot_idx - threshold_annot_shift)
                end_cut_off = min(len(p_signals[lead_idx]) - 1, last_annot_idx + threshold_annot_shift)

                mean_val = np.mean(p_signals[lead_idx])
                std_dev = 0.05

                p_signals[lead_idx, :start_cut_off] = np.random.normal(mean_val, std_dev, start_cut_off)
                p_signals[lead_idx, end_cut_off + 1:] = np.random.normal(mean_val, std_dev, max(0, len(p_signals[lead_idx]) - end_cut_off - 1))

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

        sig = np.zeros(length, dtype=np.int64)
        sig = generate_signal(symbols.p, sig, 1)
        sig = generate_signal(symbols.N, sig, 2)
        sig = generate_signal(symbols.t, sig, 3)
        return sig
    
    def parse_annotations(self, annotation: Annotation, resample_fs: int = 100) -> AnnotationSymbols:
        if not annotation.symbol:
            raise Exception("the file need to have symbols loaded")

        current_start = None
        current_peak = None
        current_symbol = None
        intervals = {"p": [], "N": [], "t": []}

        for symbol, index in zip(annotation.symbol, resample_ann(annotation.sample, annotation.fs, fs_target=resample_fs)):
            match symbol:
                case "(":
                    current_start = index
                case "p" | "N" | "t":
                    current_symbol = cast(str, symbol)
                    current_peak = index
                case ")":
                    if not (current_peak and isinstance(current_symbol, str)):
                        raise Exception("need to have peak specified before end of the interval")
                    if not current_start:
                        if not len(intervals[current_symbol]) == 0:
                            raise Exception("need to have start of the interval before parse end")
                        current_start = 0
                    intervals[current_symbol].append(Range(current_start, current_peak, index))

        return AnnotationSymbols(**intervals)

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
