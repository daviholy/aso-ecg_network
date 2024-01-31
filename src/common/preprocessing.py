import wfdb
from wfdb.processing import resample_ann, resample_sig
import plotly.graph_objs as go
from scipy import signal
import numpy as np
from pathlib import Path

def resample_record_and_annotation(record_dir: Path, record_name: str, save_dir: Path, new_fs: int=100):
    # Load the original record and annotations
    record_path = Path(record_dir) / record_name
    record_sig, record_dict = wfdb.rdrecord(str(record_path))

    # Resample each lead individually
    resampled_leads = []
    for lead_idx, lead_name in enumerate(record_dict["sig_name"]):
        resampled_lead = resample_sig(record_sig[:, lead_idx], record_dict["fs"], new_fs)
        resampled_leads.append(resampled_lead)
        annotations = wfdb.rdann(str(record_path), lead_name)
        resampled_annotations = resample_ann(annotations.sample, record_dict["fs"], new_fs)
        wfdb.wrann(f"{record_name}_{lead_name}", "", resampled_annotations, annotations.symbol, fs=new_fs, write_dir=str(save_dir))

    resampled_leads = np.array(resampled_leads).T  # Transpose to get the shape (num_samples, num_leads)

    # Save the resampled record
    save_path = Path(save_dir) / (record_name + '_resampled')
    resampled_annot = wfdb.Record(p_signal=resampled_leads, units=record_dict["units"], fs=new_fs, sig_name=record_dict["sig_name"], record_name=str(save_dir / record_name) ,n_sig=resampled_leads.shape[2])
    resampled_annot.wrsamp()
   


def draw_record():
    # Resample the ECG signal
    resampled_ecg_signal = signal.resample(record.p_signal[:, 0], int(len(record.p_signal) * target_sampling_rate / original_sampling_rate))

    # Make both signals equal in length
    min_length = min(len(resampled_ecg_signal), len(annotation_sample))
    resampled_ecg_signal = resampled_ecg_signal[:min_length]
    annotation_sample = annotation_sample[:min_length]

    # Create traces for the resampled ECG signal and annotations
    trace_resampled_ecg = go.Scatter(x=np.linspace(0, len(resampled_ecg_signal) / target_sampling_rate, len(resampled_ecg_signal)),
                                    y=resampled_ecg_signal, name=f'Resampled ECG Signal ({target_sampling_rate}Hz)')

    trace_annotations = go.Scatter(x=np.array(annotation_sample) / original_sampling_rate,
                                y=[0] * len(annotation_sample),
                                mode='markers', name='Annotations', marker=dict(symbol=annotation_symbol,
                                                                                size=8, color='red'))

    # Layout settings
    layout = go.Layout(title=f'Resampled ECG Signal with Annotations',
                    xaxis=dict(title='Time (seconds)'),
                    yaxis=dict(title='Amplitude'))

    # Create figure and plot
    fig = go.Figure(data=[trace_resampled_ecg, trace_annotations], layout=layout)

    # Show the plot
    fig.show()

if __name__ == "__main__":


    # Directory containing WFDB records
    records_directory = Path('./data/data')  # Replace with the path to your records directory
    save_directory = Path('./data/downsamp_data')

    # Target sampling rate
    target_sampling_rate = 100  # Set your desired target sampling rate

    # Iterate over files in the directory
    for file_path in records_directory.iterdir():
        if file_path.suffix == '.hea':
            # Extract the record name without extension
            record_name = file_path.stem

            # Call the resample function for each record
            resample_record_and_annotation(records_directory, record_name, save_directory, target_sampling_rate)
            # print(record_name)


