import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import Accuracy
from pathlib import Path
from common.utils import plot_jaccard_barplot

from model.ecg_network import ECGNetwork

class ECGModel(pl.LightningModule):
    def __init__(self, n_classes: int, n_channels: int, learning_rate: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ECGNetwork(n_channels, n_classes)
        self.learning_rate = learning_rate
        self.jaccard = MulticlassJaccardIndex(num_classes=n_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes) # TODO: param top_k?
        self.separate_classes_jaccard = MulticlassJaccardIndex(num_classes=n_classes, average="none")

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
    
    def on_test_epoch_start(self) -> None:
        self.classes_jaccard = {"p": [], "QRS": [], "t": []}
        return super().on_test_epoch_start()
    
    def test_step(self, batch: list[torch.Tensor], batch_idx: int):
        signal, annotation = batch
        result = self.model(signal.unsqueeze(1))

        classes_jaccard = []
        for curr_res, curr_annot in zip(result, annotation):
            classes_jaccard.append(self.separate_classes_jaccard(curr_res.permute(1, 0), curr_annot))

        for curr_classes_jaccard in classes_jaccard:
            for key, value in zip(("p", "QRS", "t"), curr_classes_jaccard):
                self.classes_jaccard[key].append(value.cpu().item())
        
    def on_test_epoch_end(self) -> None:
        # Save test dataset results as graphs
        # calculate mean
        all_jaccards = torch.tensor(list(self.classes_jaccard.values()))
        mean_jaccard = all_jaccards.mean()
        std_jaccard = all_jaccards.std()

        classes_jaccard_std = {}
        for key, value in self.classes_jaccard.items():
            self.classes_jaccard[key] = torch.tensor(value).mean().item()
            classes_jaccard_std[key] = torch.tensor(value).std().item()

        self.classes_jaccard["mean"] = mean_jaccard.item()
        classes_jaccard_std["mean"] = std_jaccard.item()

        output_path = Path("statistics/")
        if not output_path.exists():
            Path.mkdir(output_path, mode=0o775)

        plot_jaccard_barplot(self.classes_jaccard, classes_jaccard_std, output_path)
        return super().on_test_epoch_end()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
