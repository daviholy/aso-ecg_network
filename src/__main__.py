import os
from pathlib import Path

import pytorch_lightning as pl

from .Model.ecg_datamodule import ECGDataModule
from .Model.ecg_model import ECGModel

ACCELERATOR = "cpu"

# TODO: add config file
# TODO: add data augmentation
# TODO: add logger (tensorboard or aim?)
# TODO: add checkpointing and result model save

if __name__ == "__main__":
    ecg_dir = Path("data")
    n_channels = 1
    n_classes = 3
    batch_size = 32
    learning_rate = 1e-4
    num_of_workers = os.cpu_count()

    # Initialize ECG Data Module
    ecg_data_module = ECGDataModule(
        ecg_dir=ecg_dir, num_of_workers=num_of_workers, batch_size=batch_size, train_ratio=0.8
    )

    model = ECGModel(n_channels=n_channels, n_classes=n_classes, learning_rate=learning_rate)
    trainer = pl.Trainer(max_epochs=10, accelerator=ACCELERATOR)
    trainer.fit(model, ecg_data_module)
