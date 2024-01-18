import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer

from .ecg_network import ECGNetwork


class ECGModel(pl.LightningModule):
    def __init__(self, n_classes: int, n_channels: int, learning_rate: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ECGNetwork(n_channels, n_classes)
        self.learning_rate = learning_rate

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        signal, annotation = batch
        result = self.model(signal.unsqueeze(1))
        loss = F.cross_entropy(result, annotation)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        signal, annotation = batch
        result = self.model(signal.unsqueeze(1))
        loss = F.cross_entropy(result, annotation)
        self.log("val_loss", loss)
        return loss
