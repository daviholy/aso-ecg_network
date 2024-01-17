import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Optimizer, Adam

from Model.ecg_network import ECGNetwork

class ECGModel(pl.LightningModule):
    def __init__(
        self, n_channels: int, n_classes: int, learning_rate: float = 1e-4
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ECGNetwork(n_channels, n_classes)
        self.learning_rate = learning_rate

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def on_validation_epoch_end(self):
        ...