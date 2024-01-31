import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model.ecg_datamodule import ECGDataModule
from model.ecg_model import ECGModel

import plotly.graph_objects as go
import numpy as np

ACCELERATOR = "gpu"

# TODO: add config file
# TODO: make separate file for visualization?
# TODO: make test pipeline in datamodule in TensorBoard (cut start and end signal values as it is in training)

def plot_signal_with_labels(signal, labels):
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'violet'}
    fig = go.Figure()
    start_idx = 0

    fig.add_trace(go.Scatter(
    x=np.arange(len(signal)),
    y=signal,
    mode='lines',
    line=dict(color='black'),
    showlegend=False))

    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1]:
            fig.add_trace(go.Scatter(
                x=np.arange(start_idx, idx), 
                y=signal[start_idx:idx], 
                mode='lines+markers',
                line=dict(color=colors[labels[idx-1].item()]),
                showlegend=False))
            start_idx = idx

    fig.add_trace(go.Scatter(
        x=np.arange(start_idx, len(labels)), 
        y=signal[start_idx:len(labels)], 
        mode='lines+markers',
        line=dict(color=colors[labels[-1].item()]),
        showlegend=False))

    
    fig.update_layout(title='Signal Values Colored by Class',
                xaxis_title='X Axis',
                yaxis_title='Signal Value',
                legend_title='Legend')

    fig.show()


if __name__ == "__main__":
    ##################################
    train_model = False
    ##################################
    ecg_dir = Path("data")
    n_channels = 1
    n_classes = 4 # - nothing, P, QRS, T
    batch_size = 64
    learning_rate = 1e-4
    num_of_workers = os.cpu_count()
    resample_fs = 100

    # Initialize ECG Data Module
    ecg_data_module = ECGDataModule(
        ecg_dir=ecg_dir, num_of_workers=num_of_workers, batch_size=batch_size, train_ratio=0.8, resample_fs=resample_fs
    )

    tensor_board_logger = TensorBoardLogger('./logs', name='ecg_p_qrs_t')

    
    if train_model:
        model = ECGModel(n_channels=n_channels, n_classes=n_classes, learning_rate=learning_rate)
        trainer = pl.Trainer(max_epochs=200, accelerator=ACCELERATOR, logger=tensor_board_logger)
        trainer.fit(model, ecg_data_module)
    else:
        # FIXME: separate file
        model = ECGModel.load_from_checkpoint(Path("./logs/ecg_p_qrs_t/version_23/checkpoints/epoch=199-step=6000.ckpt"))
        model.to("cpu")
        ecg_data_module.setup()
        signal, target = ecg_data_module.test[1]
        pred = model.predict(signal.unsqueeze(0).unsqueeze(0)).squeeze(0)
        pred = pred.argmax(0)

        plot_signal_with_labels(signal, target)
        plot_signal_with_labels(signal, pred)     

