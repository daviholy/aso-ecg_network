from pathlib import Path
from wfdb.processing import resample_sig

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model.ecg_datamodule import ECGDataModule
from model.ecg_model import ECGModel
from model.ecg_model_inference import ModelInference

from common.config import load_config
from common.utils import plot_signal_with_labels_and_predictions

import pydicom

if __name__ == "__main__":
    config = load_config("./src/config.yaml")

    # Initialize ECG Data Module
    ecg_data_module = ECGDataModule(
        ecg_dir=config.dataloader.ecg_dir, 
        num_of_workers=config.dataloader.num_workers,
        batch_size=config.dataloader.batch_size, 
        train_ratio=config.dataloader.train_ratio, 
        resample_fs=config.dataloader.resample_fs
    )

    tensor_board_logger = TensorBoardLogger('./logs', name='ecg_p_qrs_t') if config.model.train else None

    
    trainer = pl.Trainer(max_epochs=config.training.max_epochs, 
                            accelerator=config.training.accelerator, 
                            logger=tensor_board_logger)
    
    if config.model.train:
        model = ECGModel(n_channels=config.model.n_channels, 
                         n_classes=config.model.n_classes, 
                         learning_rate=config.model.learning_rate)
        trainer.fit(model, ecg_data_module)
    else:
        model_inference = ModelInference(Path("./logs/ecg_p_qrs_t/best_model/checkpoints/epoch=199-step=6000.ckpt"))
        
        signal_path = Path('./data/our_data/david.dcm')
        dicom_data = pydicom.dcmread(signal_path)
        signals = dicom_data.waveform_array(0)
        sampling_freq = int(dicom_data.WaveformSequence[0].SamplingFrequency)

        resampled_signal = resample_sig(signals[:, 1], sampling_freq, config.dataloader.resample_fs)[0]
        sig_len_10_min = int(config.dataloader.resample_fs * 600)
        
        input_signal = resampled_signal[-sig_len_10_min:]
        input_signal = input_signal[: - config.dataloader.resample_fs * 10]

        full_pred = model_inference.process_signal(input_signal, config.dataloader.resample_fs)
        plot_signal_with_labels_and_predictions(input_signal, full_pred, save_path="./statistics", name=signal_path.stem)

        trainer.test(model_inference.model, ecg_data_module)

        signal, target = ecg_data_module.test[1+6]

        pred = model_inference.perform_inference(signal)
        model_inference.display_prediction(signal=signal, pred=pred, target=target)
