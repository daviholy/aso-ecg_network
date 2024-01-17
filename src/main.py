import pytorch_lightning as pl

from Model.ecg_model import ECGModel
from Model.ecg_datamodule import ECGDataModule

if __name__ == '__main__':
    ecg_dir = 'data/dataset'
    n_channels = 12
    n_classes = 3
    batch_size = 32
    learning_rate = 1e-4

    # Initialize ECG Data Module
    ecg_data_module = ECGDataModule(
        ecg_dir=ecg_dir,
        batch_size=batch_size,
        num_of_workers=8,
        train_ratio=0.8
    )

    model = ECGModel(n_channels=n_channels, n_classes=n_classes, learning_rate=learning_rate)
    trainer = pl.Trainer(max_epochs=10, accelerator="cuda")
    trainer.fit(model, ecg_data_module)