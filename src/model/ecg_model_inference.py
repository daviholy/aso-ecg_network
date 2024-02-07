from pathlib import Path
from typing import Any
import torch

from common.utils import plot_signal_with_labels_and_predictions
from model.ecg_model import ECGModel


class ModelInference:
    def __init__(
        self, path_to_model: Path, device: str = "cpu"
    ) -> None:
        self.model = self.load_model(path_to_model)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def perform_inference(self, ecg_sig: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # check if ecg_sig has shape [batch, n_channels, 1000]
            while len(ecg_sig.shape) < 3:
                ecg_sig = ecg_sig.unsqueeze(0)

            ecg_sig.to(self.device)
            prediction: torch.Tensor = self.model(ecg_sig)
        return prediction.squeeze(0).argmax(0).cpu()

    def display_prediction(self, signal: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
        plot_signal_with_labels_and_predictions(signal=signal, pred=pred, target=target)

    def process_signal(self, signal, resampling_rate): # TODO: move to model_inference

        # Calculate the number of samples in a 10-second window
        window_size = 10 * resampling_rate

        min_val = signal.min()
        max_val = signal.max()
        normalized_signal = 2 * ((signal - min_val) / (max_val - min_val)) - 1
        
        with torch.no_grad():
            predictions = []
            # Process each 10-second window
            for start_idx in range(0, len(signal), window_size):
                end_idx = start_idx + window_size

                window = normalized_signal[start_idx:end_idx]
                
                prediction = self.perform_inference(torch.from_numpy(window).float())
                predictions.append(prediction)
            
            full_prediction = torch.cat(predictions, dim=0)

        return full_prediction

    @staticmethod
    def load_model(path_to_model: Path) -> Any:
        return ECGModel.load_from_checkpoint(path_to_model)
