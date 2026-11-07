"""
This script provides a full pipeline for processing neurophysiological data recorded using
Neuronexus probes. It is designed to load raw signal data, preprocess it, and extract
features for machine learning applications, particularly for brain region decoding.

The pipeline consists of several key steps:
1.  **Data Loading**: It loads raw electrophysiological signals (either wide-band or LFP)
    and corresponding channel labels from various file formats (.dat, .h5, .json, .xml).
    It uses helper functions from the `blind_localization` package.

2.  **Signal Preprocessing**: The raw signals are processed through several steps:
    -   Filtering (low-pass and high-pass) to isolate frequencies of interest and remove noise.
    -   Unit conversion from ADC values to volts, using a specific conversion factor.

3.  **Feature Extraction**: For each channel, the script segments the data into trials
    based on a predefined set of start times. For each trial, it extracts two types
    of features:
    a.  **Spectrograms**: Time-frequency representations are computed from the LFP signal.
    b.  **PSD-based Features**: Power is calculated for different neurophysiological bands
        (delta, theta, alpha, beta, gamma) from the wide-band raw signal.

4.  **Data Saving**: The extracted features (spectrograms, PSD features) and the raw
    signal chunks are saved into separate pickle files for each recording session.

Two main pipelines are defined:
-   `preprocess_data`: Processes data with known ground-truth labels.
-   `preprocess_signal_pipeline`: Processes data without labels, assigning a dummy label.

NOTE: This script contains numerous hardcoded paths specific to a particular HPC environment.
These will need to be modified for use in a different setting.
"""

import os
import sys
import pickle
import h5py
import numpy as np
import scipy.io
from tqdm import tqdm

# --- Local and Project-specific Imports ---
# NOTE: The following imports rely on a specific project structure and paths.
# The sys.path.append might need adjustment based on the execution environment.
sys.path.append(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding'))
from script.utils import compute_spectrogram_librosa, calculate_rms, calculate_power
from blind_localization.data.data_loading import load_session_data
from blind_localization.data.preprocess import process_labels, process_signals, load_data
from script.ibl_preprocessing.preprocess_psd_spectrogram import preprocess_signal


def load_signal_labels(source_session, sample_rate=20000, T=30, data_type='lfp'):
    """
    Loads and preprocesses signal data and corresponding labels for a given session.

    Args:
        source_session (str): The name of the session to load.
        sample_rate (int): The sampling rate of the signal.
        T (int): The total duration of the signal to load in seconds.
        data_type (str): The type of data to load ('lfp' or 'raw').

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The preprocessed signal data (channels x samples).
            - np.ndarray: The final class labels for each channel.
            - np.ndarray: The timestamps for sharp-wave ripples (SWRs).
    """
    # --- Load File Paths ---
    # NOTE: Path to file_path_hpc.json is hardcoded.
    file_path_config = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/script/Neuronexus_preprocessing/file_path_hpc.json')
    source_file = load_session_data(file_path_config, source_session)
    public_file = load_session_data(file_path_config, "public")
    signal_path = source_file["raw_lfp_path"] if data_type == 'lfp' else source_file["raw_signal_path"]
    print(f"Loading signal from: {signal_path}")

    # --- Load and Process Raw Data and Labels ---
    raw_signal, df, skipped_channels = load_data(
        signal_path, public_file["label_path"], source_file["xml_path"],
        sheet_name=source_file["sheet_name"], sample_rate=sample_rate, T=T
    )
    channel_region_map, skipped_channels, channel_channel_map = process_labels(
        df, public_file["mapping_path"], skipped_channels
    )
    raw_signal = process_signals(raw_signal, channel_channel_map, normalize=False)
    
    # --- Preprocess and Convert Signal ---
    # Apply filtering (high-pass, optional low-pass)
    raw_signal = preprocess_signal(raw_signal, original_sampling_rate=sample_rate, target_sampling_rate=sample_rate, low_pass=True)
    # Convert ADC units to volts (specific to Intan acquisition systems)
    raw_signal = raw_signal * 0.195 / 1000 / 1000

    # --- Load and Remap Channel Labels ---
    # NOTE: Path to labels is hardcoded.
    channel_labels = np.load(f"/workspace/ws5/region_decoding/data/Neuronexus/labels_{source_session}.npy")
    labels = np.argmax(channel_labels, axis=1) + 1
    channel_labels = channel_labels * labels[:, np.newaxis]
    class_labels = np.sum(channel_labels, axis=1).astype(int)
    
    # Remap numerical labels to a new scheme. This can be for consistency or to match model outputs.
    remap_dict_1 = {0:5, 1:0, 2:1, 3:2, 4:3, 5:4}
    class_labels = np.array([remap_dict_1[c] for c in class_labels])
    remap_dict_2 = {0:4, 1:0, 2:1, 3:2, 4:3, 5:5}
    class_labels = np.array([remap_dict_2[c] for c in class_labels])

    # --- Load SWR Timestamps ---
    swr_file = scipy.io.loadmat(source_file["swr_path"])
    swr_timestamp = swr_file['ripples']['timestamps'][0][0]
  
    return raw_signal, class_labels, swr_timestamp


def load_signal_pipeline(raw_signal_path, sample_rate=20000, T=30):
    """
    Loads and preprocesses a raw signal from an HDF5 file.

    Args:
        raw_signal_path (str): Path to the HDF5 file containing the signal.
        sample_rate (int): The sampling rate of the signal.
        T (int): The duration of the signal to load in seconds.

    Returns:
        np.ndarray: The preprocessed signal data.
    """
    with h5py.File(raw_signal_path, 'r') as file:
        raw_signal = file['data'][:, :int(sample_rate * T)]

    raw_signal = preprocess_signal(raw_signal, original_sampling_rate=sample_rate, target_sampling_rate=sample_rate, low_pass=True)
    raw_signal = raw_signal * 0.195 / 1000 / 1000  # Convert ADC units to volts
    
    return raw_signal


def preprocess_signal_pipeline(sessions, session_config, data_type='lfp'):
    """
    A pipeline to process signals that do not have associated ground-truth labels.
    It assigns a dummy label (0) to all extracted features.

    Args:
        sessions (list): A list of session names to process.
        session_config (dict): A dictionary containing configuration for the processing.
        data_type (str): The type of signal to process ('lfp' or 'raw').
    """
    t_start = session_config['t_starts'][0]
    dT = session_config['trial_length']

    for source_session in sessions:
        # Calculate total session length from the trial start times
        T = session_config['t_starts'][-1] - session_config['t_starts'][0] + (session_config['t_starts'][1] - session_config['t_starts'][0])
        print(f"Processing session: {source_session}, Total length: {T} sec")

        # --- Setup Paths ---
        file_path_config = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/script/Neuronexus_preprocessing/file_path_hpc.json')
        source_file = load_session_data(file_path_config, source_session)
        signal_path = source_file["raw_lfp_path"] if data_type == 'lfp' else source_file["raw_signal_path"]
        print(f"Loading signal from: {signal_path}")

        folder_path = session_config['folder_path']
        file_path_spectrogram = os.path.join(folder_path, 'spectrogram')
        file_path_raw = os.path.join(folder_path, 'raw')
        file_path_lfp = os.path.join(folder_path, 'lfp')
        os.makedirs(file_path_spectrogram, exist_ok=True)
        os.makedirs(file_path_raw, exist_ok=True)
        os.makedirs(file_path_lfp, exist_ok=True)

        # --- Load and Process Data ---
        lfp_signal = load_signal_pipeline(signal_path, sample_rate=1250, T=T).astype(np.float32)
        print(f"LFP signal shape: {lfp_signal.shape}")

        data_list, feature_list, lfp_list = [], [], []

        # Iterate over each channel
        for i, channel_data in tqdm(enumerate(lfp_signal), total=len(lfp_signal), desc="Processing Channels"):
            lfp_channel_data = lfp_signal[i]
            label = 0  # Assign a dummy label

            # Iterate over each trial defined by start times
            for count, ts in enumerate(session_config['t_starts']):
                # --- Extract Spectrogram from LFP ---
                ts_start_lfp, ts_end_lfp = int((ts - t_start) * 1250), int((ts - t_start + dT) * 1250)
                row_lfp = lfp_channel_data[ts_start_lfp:ts_end_lfp]
                z_spect = compute_spectrogram_librosa(row_lfp, 1250, num_freq_bins=500, time_bins=16, normalizing='zscore')
                if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                    data_list.append((z_spect, label, count, i))
                    lfp_list.append((row_lfp, label, count, i))

                # --- Extract PSD features from Raw Wide-band Signal ---
                ts_start_raw, ts_end_raw = int((ts - t_start) * 20000), int((ts - t_start + dT) * 20000)
                row_raw = channel_data[ts_start_raw:ts_end_raw]
                if not np.any(np.isnan(row_raw)) and not np.any(np.isinf(row_raw)):
                    features = [
                        calculate_rms(row_raw),
                        *calculate_power(row_raw, 20000),          # Whole band
                        *calculate_power(row_raw, 20000, (0, 4)),   # Delta
                        *calculate_power(row_raw, 20000, (4, 8)),   # Theta
                        *calculate_power(row_raw, 20000, (8, 12)),  # Alpha
                        *calculate_power(row_raw, 20000, (12, 30)), # Beta
                        *calculate_power(row_raw, 20000, (30, -1))  # Gamma
                    ]
                    feature_list.append((features, label, count, i))

        # --- Save Extracted Data ---
        with open(os.path.join(file_path_spectrogram, f"{source_session}_data.pickle"), 'wb') as f:
            pickle.dump(data_list, f)
            print(f"Saved {len(data_list)} spectrograms to {f.name}")
        with open(os.path.join(file_path_raw, f"{source_session}_data.pickle"), 'wb') as f:
            pickle.dump(feature_list, f)
            print(f"Saved {len(feature_list)} PSD features to {f.name}")
        with open(os.path.join(file_path_lfp, f"{source_session}_lfp.pickle"), 'wb') as f:
            pickle.dump(lfp_list, f)
            print(f"Saved {len(lfp_list)} LFP chunks to {f.name}")


def preprocess_data(sessions, session_config):
    """
    A pipeline to process signals with known ground-truth labels.

    Args:
        sessions (list): A list of session names to process.
        session_config (dict): A dictionary containing configuration for the processing.
    """
    t_start = session_config['t_starts'][0]
    dT = session_config['trial_length']
    region_counts = [0] * 6  # Assuming 6 possible region labels

    for source_session in sessions:
        T = session_config['t_starts'][-1] - session_config['t_starts'][0] + (session_config['t_starts'][1] - session_config['t_starts'][0])
        print(f"Processing session: {source_session}, Total length: {T} sec")

        # --- Setup Paths ---
        folder_path = session_config['folder_path']
        file_path_spectrogram = os.path.join(folder_path, 'spectrogram')
        file_path_raw = os.path.join(folder_path, 'raw')
        file_path_lfp = os.path.join(folder_path, 'lfp')
        os.makedirs(file_path_spectrogram, exist_ok=True)
        os.makedirs(file_path_raw, exist_ok=True)
        os.makedirs(file_path_lfp, exist_ok=True)

        # --- Load LFP, Raw Signal, and Labels ---
        lfp_signal, channel_labels, _ = load_signal_labels(source_session, sample_rate=1250, T=T, data_type='lfp')
        raw_signal, _, _ = load_signal_labels(source_session, sample_rate=20000, T=T, data_type='raw')
        print(f"LFP signal shape: {lfp_signal.shape}, Raw signal shape: {raw_signal.shape}")

        data_list, raw_list, feature_list, lfp_list = [], [], [], []

        # Iterate over each channel and its corresponding label
        for i, (channel_data, label) in tqdm(enumerate(zip(raw_signal, channel_labels)), total=len(raw_signal), desc="Processing Channels"):
            # Skip channels with a specific label (e.g., 5 might be an invalid or unwanted region)
            if label == 5:
                continue
            
            lfp_channel_data = lfp_signal[i]

            # Iterate over each trial
            for count, ts in enumerate(session_config['t_starts']):
                # --- Extract Spectrogram from LFP ---
                ts_start_lfp, ts_end_lfp = int((ts - t_start) * 1250), int((ts - t_start + dT) * 1250)
                row_lfp = lfp_channel_data[ts_start_lfp:ts_end_lfp]
                z_spect = compute_spectrogram_librosa(row_lfp, 1250, num_freq_bins=500, time_bins=16, normalizing='zscore')
                if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                    data_list.append((z_spect, label, count, i))
                    lfp_list.append((row_lfp, label, count, i))
                    region_counts[label] += 1

                # --- Extract PSD features from Raw Wide-band Signal ---
                ts_start_raw, ts_end_raw = int((ts - t_start) * 20000), int((ts - t_start + dT) * 20000)
                row_raw = channel_data[ts_start_raw:ts_end_raw]
                if not np.any(np.isnan(row_raw)) and not np.any(np.isinf(row_raw)):
                    raw_list.append((row_raw, label, count, i))
                    features = [
                        calculate_rms(row_raw),
                        *calculate_power(row_raw, 20000),
                        *calculate_power(row_raw, 20000, (0, 4)),
                        *calculate_power(row_raw, 20000, (4, 8)),
                        *calculate_power(row_raw, 20000, (8, 12)),
                        *calculate_power(row_raw, 20000, (12, 30)),
                        *calculate_power(row_raw, 20000, (30, -1))
                    ]
                    feature_list.append((features, label, count, i))

        # --- Save Extracted Data ---
        with open(os.path.join(file_path_spectrogram, f"{source_session}_data.pickle"), 'wb') as f:
            pickle.dump(data_list, f)
            print(f"Saved {len(data_list)} spectrograms to {f.name}")
        with open(os.path.join(file_path_lfp, f"{source_session}_raw.pickle"), 'wb') as f:
            pickle.dump(raw_list, f)
            print(f"Saved {len(raw_list)} raw signal chunks to {f.name}")
        with open(os.path.join(file_path_raw, f"{source_session}_data.pickle"), 'wb') as f:
            pickle.dump(feature_list, f)
            print(f"Saved {len(feature_list)} PSD features to {f.name}")
        with open(os.path.join(file_path_lfp, f"{source_session}_lfp.pickle"), 'wb') as f:
            pickle.dump(lfp_list, f)
            print(f"Saved {len(lfp_list)} LFP chunks to {f.name}")


def main():
    """Main entry point for the script."""
    # --- Configuration ---
    # NOTE: The folder_path is hardcoded.
    folder_path = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/Neuronexus')
    session_config = {
        "visualize": True,  # Flag for visualization (not used in this script)
        'swr_flag': False,  # Flag for SWR analysis (not used in this script)
        't_starts': np.arange(2160, 2340, 3),  # Defines start times for 3-sec trials
        'trial_length': 3,  # Length of each trial in seconds
        'folder_path': folder_path
    }

    # --- Execution ---
    # The user can choose which pipeline to run and for which sessions.
    
    # Example for running the labeled data pipeline:
    # session_names_labeled = ["AD_HF01_1", "AD_HF02_2", "AD_HF02_4", "AD_HF03_1", "AD_HF03_2", "NN_syn_01", "NN_syn_02"]
    # preprocess_data(session_names_labeled, session_config)

    # Example for running the unlabeled data pipeline:
    session_names_unlabeled = ["NN_syn_03"]
    preprocess_signal_pipeline(session_names_unlabeled, session_config)


if __name__ == "__main__":
    main()

