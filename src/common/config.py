from pathlib import Path
import yaml
from pathlib import Path
from pydantic import BaseModel


class ModelConfig(BaseModel):
    n_classes: int
    n_channels: int
    learning_rate: float
    train: bool


class DataLoaderConfig(BaseModel):
    ecg_dir: Path
    batch_size: int
    num_workers: int
    train_ratio: float
    resample_fs: int


class TrainingConfig(BaseModel):
    accelerator: str
    max_epochs: int


class Config(BaseModel):
    model: ModelConfig
    dataloader: DataLoaderConfig
    training: TrainingConfig


def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(**config_dict)
