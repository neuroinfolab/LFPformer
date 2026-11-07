"""
This script serves as a comprehensive pipeline for preprocessing raw LFP data from the IBL
and then extracting two types of features: spectrograms and Power Spectral Density (PSD) based features.

The pipeline is as follows:
1.  **Load Data**: For each experimental session (EID), it loads the raw LFP signal,
    sampling rate, and channel labels from .npy files.
2.  **Signal Preprocessing**:
    -   Temporally downsamples the signal to a target sampling rate (e.g., 1250 Hz).
    -   Applies a low-pass anti-aliasing filter before downsampling.
    -   Applies a high-pass filter to remove low-frequency drift.
3.  **Feature Extraction**:
    -   It iterates through channels in specified brain regions (e.g., CA1, VIS).
    -   The preprocessed signal for each channel is segmented into smaller chunks.
    -   For each chunk, it computes:
        a.  A **spectrogram**, which is a time-frequency representation of the signal.
        b.  A set of **PSD-based features**, including RMS and power in different
            neurophysiological frequency bands (delta, theta, alpha, beta, gamma).
4.  **Save Data**: The script saves three separate files for each session:
    -   A pickle file containing the computed spectrograms.
    -   A pickle file containing the raw LFP chunks corresponding to the spectrograms.
    -   A pickle file containing the computed PSD-based features.

The script is configurable via command-line arguments and contains hardcoded paths that
may need to be adjusted for different environments.
"""

import os
import sys
import time
import pickle
import argparse
import warnings
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import gc

# Add the parent directory to the system path to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from script.utils import compute_spectrogram_librosa, calculate_rms, calculate_power

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

def parse_arguments():
    """
    Parses command-line arguments for the data processing pipeline.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Parse arguments for LFP data preprocessing and feature extraction.")
    parser.add_argument('--chunk_size', type=int, default=3, help="Size of each data chunk in seconds.")
    parser.add_argument('--start_time', type=float, default=30, help="Start time for the analysis in seconds.")
    parser.add_argument('--n_chunks', type=int, default=100, help="Number of chunks to process per channel.")
    parser.add_argument('--start_eid_idx', type=int, default=0, help="Start index of the EID list to process.")
    parser.add_argument('--end_eid_idx', type=int, default=7, help="End index of the EID list to process.")
    return parser.parse_args()

def preprocess_signal(data, original_sampling_rate, target_sampling_rate, low_pass=False):
    """
    Preprocesses the raw LFP signal by downsampling and filtering.

    Args:
        data (np.ndarray): 2D numpy array (channels x samples).
        original_sampling_rate (float): Original sampling rate in Hz.
        target_sampling_rate (float): Target sampling rate in Hz.
        low_pass (bool): Whether to apply an additional low-pass filter at 500 Hz.

    Returns:
        np.ndarray: 2D numpy array of preprocessed data.
    """
    print("Starting signal preprocessing...")
    start_t = time.time()

    # 1. Temporal Downsampling
    downsample_factor = int(original_sampling_rate / target_sampling_rate)
    if downsample_factor > 1:
        print(f"Downsampling temporally by a factor of {downsample_factor}.")
        # Apply an anti-aliasing filter before downsampling to prevent aliasing
        nyquist_original = 0.5 * original_sampling_rate
        cutoff = 0.5 * target_sampling_rate  # Cutoff should be Nyquist of the target rate
        sos = signal.butter(4, cutoff / nyquist_original, btype='low', output='sos')
        data = signal.sosfiltfilt(sos, data, axis=1)
        data = data[:, ::downsample_factor]
    else:
        print("No temporal downsampling applied as target rate is similar to original.")

    # 2. Low-pass Filtering at 500 Hz (optional)
    if low_pass:
        print("Applying low-pass filter at 500 Hz.")
        nyquist_new = 0.5 * target_sampling_rate
        sos_lp = signal.butter(2, 500 / nyquist_new, btype='lowpass', output='sos')
        data = signal.sosfiltfilt(sos_lp, data, axis=1)

    # 3. High-pass Filtering at 0.1 Hz to remove slow drifts
    print("Applying high-pass filter at 0.1 Hz.")
    nyquist_new = 0.5 * target_sampling_rate
    sos_hp = signal.butter(2, 0.1 / nyquist_new, btype='highpass', output='sos')
    data = signal.sosfiltfilt(sos_hp, data, axis=1)

    # Note: Common average re-referencing is commented out but can be enabled if needed.
    # print("Applying common average re-referencing.")
    # reference = np.mean(data, axis=0)
    # data = data - reference

    print(f"Preprocessing finished in {time.time() - start_t:.2f} seconds.")
    return data

def load_session(eid, data_path):
    """
    Loads the LFP data, sampling rate, and channel labels for a given EID.

    Args:
        eid (str): The experiment ID.
        data_path (str): The path to the directory containing the data files.

    Returns:
        tuple: A tuple containing (lfp_data, sampling_rate, labels).
    """
    lfp_data = np.load(os.path.join(data_path, f'{eid}_lfp.npy'))
    sampling_rate = np.load(os.path.join(data_path, f'{eid}_fs.npy'))
    labels = np.load(os.path.join(data_path, f'{eid}_label_lfp.npy'), allow_pickle=True)
    return lfp_data, sampling_rate, labels

def preprocess_and_extract_features(chunk_size, start_time, n_chunks, start_eid_idx, end_eid_idx):
    """
    Main function to run the preprocessing and feature extraction pipeline.

    Args:
        chunk_size (int): The size of each data chunk in seconds.
        start_time (float): The start time in seconds for data extraction.
        n_chunks (int): The number of chunks to process for each channel.
        start_eid_idx (int): The starting index from the filtered sessions list.
        end_eid_idx (int): The ending index from the filtered sessions list.
    """
    print("Job start: Preprocessing and extracting features.")
    print(f"Current working directory: {os.getcwd()}")

    # --- Setup ---
    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'VIS'}
    acr_dict = {acr: i for i, acr in enumerate(sorted(list(hc_acronyms)))}
    print(f"Region to ID mapping: {acr_dict}")

    # --- Paths and Data Loading ---
    # NOTE: These paths are hardcoded and may need to be changed.
    input_path = '/vast/ws4/data/ibl_new'
    output_path_spectrogram = "/vast/ws4/data/ibl_new/spectrogram_preprocessed"
    output_path_psd = "/vast/ws4/data/ibl_new/raw_preprocessed"
    mat_output_path = __import__('lfpformer.paths',fromlist=['legacy_path']).legacy_path('/workspace/ws4/region_decoding/data/ibl/raw_signal_mat/')

    os.makedirs(output_path_spectrogram, exist_ok=True)
    os.makedirs(output_path_psd, exist_ok=True)
    os.makedirs(mat_output_path, exist_ok=True)

    # Discover sessions from the files in the input directory
    filtered_sessions = sorted(list(set(
        '_'.join(p.split('_')[:2]) for p in os.listdir(input_path)
        if not os.path.isdir(os.path.join(input_path, p)))))
    
    print(f"Total number of sessions found: {len(filtered_sessions)}")
    sessions_to_process = filtered_sessions[start_eid_idx:end_eid_idx]
    print(f"Processing {len(sessions_to_process)} sessions from index {start_eid_idx} to {end_eid_idx}.")

    # --- Main Processing Loop ---
    for eid in sessions_to_process:
        # Check if output already exists to make the script resumable
        if os.path.exists(os.path.join(output_path_psd, f"{eid}_data.pickle")):
            print(f"Skipping session {eid} as output already exists.")
            continue

        print(f"\n--- Processing Session: {eid} ---")
        try:
            session_data, sampling_rate, labels = load_session(eid, input_path)
        except FileNotFoundError:
            print(f"Data for session {eid} not found. Skipping.")
            continue

        time_in_seconds = session_data.shape[1] / sampling_rate
        print(f"Session {eid} has {time_in_seconds:.2f} seconds of data at {sampling_rate} Hz.")

        # Save a snippet of the raw signal to a .mat file for external analysis (e.g., MATLAB)
        mat_file_path = os.path.join(mat_output_path, f"raw_{eid}.mat")
        savemat(mat_file_path, {"lfp_signal": session_data[:, :500*2500], "labels": labels})
        print(f"Saved raw signal snippet to {mat_file_path}")

        # --- Preprocess Signal ---
        target_sampling_rate = 1250
        preprocessed_data = preprocess_signal(session_data, sampling_rate, target_sampling_rate)
        sampling_rate = target_sampling_rate  # Update sampling rate for subsequent calculations

        # --- Feature Extraction ---
        spectrogram_list, raw_list, psd_list = [], [], []
        region_counts = [0] * len(acr_dict)

        for region in hc_acronyms:
            chan_ids = [i for i, label in enumerate(labels) if region in label]
            if not chan_ids:
                continue
            
            regional_lfp_data = preprocessed_data[chan_ids]
            print(f"Region '{region}' has {len(chan_ids)} channels. Extracting features...")

            for chan_id, channel_data in zip(chan_ids, regional_lfp_data):
                window_size = int(chunk_size * sampling_rate)
                start = int(start_time * sampling_rate)
                end = start + window_size
                count = 0
                
                while count < n_chunks and end < len(channel_data):
                    row = channel_data[start:end]
                    if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                        # 1. Compute Spectrogram
                        z_spect = compute_spectrogram_librosa(row, sampling_rate, num_freq_bins=500, time_bins=16, normalizing='zscore')
                        
                        if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                            y = acr_dict[region]
                            spectrogram_list.append((z_spect, y, count, chan_id))
                            raw_list.append((row, y, count, chan_id))
                            
                            # 2. Compute PSD-based features
                            features = [
                                calculate_rms(row),
                                *calculate_power(row, sampling_rate),          # Whole band
                                *calculate_power(row, sampling_rate, (0, 4)),   # Delta
                                *calculate_power(row, sampling_rate, (4, 8)),   # Theta
                                *calculate_power(row, sampling_rate, (8, 12)),  # Alpha
                                *calculate_power(row, sampling_rate, (12, 30)), # Beta
                                *calculate_power(row, sampling_rate, (30, -1))  # Gamma
                            ]
                            psd_list.append((features, y, count, chan_id))
                            
                            count += 1
                            region_counts[y] += 1

                    start += window_size
                    end += window_size

        print(f"Total data points created for session {eid}: {len(spectrogram_list)}")
        print(f"Region counts: {region_counts}")

        # --- Save Processed Data ---
        if spectrogram_list:
            with open(os.path.join(output_path_spectrogram, f"{eid}_data.pickle"), 'wb') as f:
                pickle.dump(spectrogram_list, f)
                print(f"Saved {len(spectrogram_list)} spectrograms to {f.name}")
            with open(os.path.join(output_path_spectrogram, f"{eid}_raw.pickle"), 'wb') as f:
                pickle.dump(raw_list, f)
                print(f"Saved {len(raw_list)} raw chunks to {f.name}")
            with open(os.path.join(output_path_psd, f"{eid}_data.pickle"), 'wb') as f:
                pickle.dump(psd_list, f)
                print(f"Saved {len(psd_list)} PSD features to {f.name}")
        
        # Clean up memory
        del session_data, preprocessed_data, spectrogram_list, raw_list, psd_list
        gc.collect()

    print("\nAll sessions processed.")


if __name__ == "__main__":
    args = parse_arguments()
    preprocess_and_extract_features(
        chunk_size=args.chunk_size,
        start_time=args.start_time,
        n_chunks=args.n_chunks,
        start_eid_idx=args.start_eid_idx,
        end_eid_idx=args.end_eid_idx
    )
    print("Job finished.")
