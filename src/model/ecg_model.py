import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import Accuracy

from .ecg_network import ECGNetwork

class ECGModel(pl.LightningModule):
    def __init__(self, n_classes: int, n_channels: int, learning_rate: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ECGNetwork(n_channels, n_classes)
        self.learning_rate = learning_rate
        self.jaccard = MulticlassJaccardIndex(num_classes=n_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes) # TODO: param top_k?

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        signal, annotation = batch
        result = self.model(signal.unsqueeze(1))
        loss = F.cross_entropy(result, annotation)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        signal, annotation = batch
        result = self.model(signal.unsqueeze(1))
        loss = F.cross_entropy(result, annotation)
        jaccard_index = self.jaccard(result, annotation)
        acc = self.accuracy(result, annotation)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_jaccard_index", jaccard_index, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True)
        return loss
