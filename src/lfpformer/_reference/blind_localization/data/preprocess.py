import json
import os

import h5py
import numpy as np
import pandas as pd
from lxml import etree
from scipy.signal import butter, filtfilt
from spikeinterface import NumpyRecording
from spikeinterface.preprocessing import bandpass_filter
from tqdm import tqdm


def load_large_signal(raw_signal_path, sample_rate=20000, T=30, downsample_factor=4):
    """
    Loads a large signal from an HDF5 file, with an option to downsample in chunks.

    If a downsample_factor is provided, the function processes the signal in chunks,
    downsamples each chunk, and saves the result to a .npy file for faster loading next time.
    If not downsampling, it loads only the first T seconds of the signal.

    Args:
        raw_signal_path (str): The path to the HDF5 file containing the raw signal.
        sample_rate (int): The sampling rate of the signal.
        T (int): The duration in seconds to load if not downsampling.
        downsample_factor (int, optional): The factor by which to downsample the signal.

    Returns:
        np.ndarray: The loaded (and possibly downsampled) signal data.
    """
    with h5py.File(raw_signal_path, 'r') as file:
        raw_signal = file['data']

        if downsample_factor:
            # If downsampling, check if a pre-processed .npy file already exists.
            processed_path = os.path.splitext(raw_signal_path)[0] + '.npy'
            if os.path.isfile(processed_path):
                print(f"Loading pre-processed signal from {processed_path}")
                resampled_signal = np.load(processed_path)
            else:
                # If not pre-processed, downsample the signal in chunks to manage memory.
                print("Downsampling signal in chunks...")
                n_channel, n_timepoints = raw_signal.shape
                n_timepoints_resampled = n_timepoints // downsample_factor

                resampled_signal = np.zeros((n_channel, n_timepoints_resampled))
                # Process the signal one second at a time.
                for start in tqdm(range(0, n_timepoints, sample_rate)):
                    end = min(start + sample_rate, n_timepoints)
                    chunk = raw_signal[:, start:end]

                    for i in range(n_channel):
                        resampled_chunk = downsample(chunk[i], sample_rate, downsample_factor)
                        resampled_signal[i, start // downsample_factor:end // downsample_factor] = resampled_chunk
                
                # Save the downsampled signal for future use.
                np.save(processed_path, resampled_signal)
            return resampled_signal
        
        # If not downsampling, load the first T seconds of the raw signal.
        print(f"Loading first {T} seconds of raw signal.")
        return raw_signal[:, :int(sample_rate * T)]

def downsample(raw_signal, fs_source, downsampling_factor, fs_target=1250):
    """
    Applies a low-pass Butterworth filter and then downsamples the signal.

    Args:
        raw_signal (np.ndarray): The 1D signal to downsample.
        fs_source (int): The source sampling frequency.
        downsampling_factor (int): The factor to downsample by.
        fs_target (int): The target sampling frequency, used to calculate the filter cutoff.

    Returns:
        np.ndarray: The downsampled signal.
    """
    # Design a low-pass Butterworth filter to prevent aliasing.
    fs_target_nyquist = (fs_target / downsampling_factor) / 2
    b, a = butter(5, fs_target_nyquist / (fs_source / 2), btype='low')
    filtered_signal = filtfilt(b, a, raw_signal)

    # Downsample by taking every N-th sample.
    downsampled_signal = filtered_signal[::downsampling_factor]
    return downsampled_signal


def load_data(raw_signal_path, label_path, xml_path, sheet_name, sample_rate=20000, T=30):
    """
    Loads raw signal data, region labels, and information about skipped channels.

    Args:
        raw_signal_path (str): Path to the HDF5 signal file.
        label_path (str): Path to the Excel file with region labels.
        xml_path (str): Path to the XML file describing channel properties.
        sheet_name (str): The name of the sheet to read from in the Excel file.
        sample_rate (int): The sampling rate of the signal.
        T (int): The duration in seconds to load from the signal file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The raw signal data.
            - pd.DataFrame: The DataFrame with region labels.
            - np.ndarray: An array of channel indices that should be skipped.
    """
    # Load raw data from the HDF5 file.
    raw_signal = load_large_signal(raw_signal_path, sample_rate, T, downsample_factor=None)
    
    # Load region labels from the specified Excel sheet.
    df = pd.read_excel(label_path, sheet_name=sheet_name)

    # Parse the XML file to identify channels marked to be skipped.
    tree = etree.parse(xml_path)
    root = tree.getroot()
    element = root.find('anatomicalDescription')

    skip_arr = np.zeros((1024,))
    for channel in element.findall('.//channel'):
        skip_value = channel.attrib['skip']
        c = int(channel.text)
        skip_arr[c] = skip_value

    skipped_channels = np.where(skip_arr == 1)[0]

    return raw_signal, df, skipped_channels


def process_labels(df, mapping_path, skipped_channels):
    """
    Cleans and standardizes the region labels and channel numbers.

    This function handles missing values, removes bad channels, standardizes region names,
    and corrects the channel ordering based on the probe's physical layout.

    Args:
        df (pd.DataFrame): The DataFrame with raw channel and region data.
        mapping_path (str): Path to a JSON file for mapping region names.
        skipped_channels (np.ndarray): An array of channel indices to be removed.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The processed DataFrame with corrected channels and labels.
            - np.ndarray: The updated array of skipped channel indices (after re-mapping).
            - dict: A dictionary mapping original channel indices to corrected indices.
    """
    with open(mapping_path) as json_file:
        mapping = json.load(json_file)
    
    # Combine the interleaved channel and region columns into two series.
    channels = pd.concat([df.iloc[:, i] for i in range(0, len(df.columns), 2)], ignore_index=True)
    regions = pd.concat([df.iloc[:, i+1] for i in range(0, len(df.columns), 2)], ignore_index=True)

    output = pd.DataFrame({"channels": channels, "regions": regions})
    output["channels"] -= 1  # Adjust to be 0-indexed.

    # Fill in NaN values.
    for idx, row in output.iterrows():
        if pd.isna(row["channels"]):
            output.at[idx, "channels"] = output.at[idx-1, "channels"] - 1
        if pd.isna(row["regions"]):
            output.at[idx, "regions"] = "UNK"  # Mark unknown regions.

    # Remove channels that are marked as skipped.
    mask = ~output["channels"].isin(skipped_channels)
    output = output[mask]

    # Standardize region names using the provided mapping.
    output = output.replace(mapping)

    # Correct the channel numbering to match the physical probe layout.
    # The original numbering is shuffled; this creates a linear mapping.
    corrected_channels = np.array([list(range(shank*128+127, shank*128-1, -1)) for shank in range(8)]).flatten()
    channel_channel_map = dict(zip(processed_channels, corrected_channels))
    output = output.replace(channel_channel_map)

    output["channels"] = output["channels"].astype(int)

    # Update the skipped_channels list to reflect the corrected channel numbers.
    vectorized_func = np.vectorize(channel_channel_map.get)
    skipped_channels = vectorized_func(skipped_channels) if len(skipped_channels) > 0 else skipped_channels

    return output, skipped_channels, channel_channel_map


def process_signals(raw_signal, channel_channel_map, normalize=True):
    """
    Reorders and optionally normalizes the raw signal channels.

    Args:
        raw_signal (np.ndarray): The input signal array (n_channels, n_samples).
        channel_channel_map (dict): A map from original channel index to corrected index.
        normalize (bool): If True, z-score normalizes each channel.

    Returns:
        np.ndarray: The processed signal array.
    """
    raw_signal = raw_signal.astype('float32')
    processed_signal = np.zeros_like(raw_signal)

    # Apply the channel reordering and normalization.
    for i in tqdm(range(len(raw_signal))):
        j = channel_channel_map.get(i) # Use .get() to avoid errors if a channel is missing
        if j is None:
            continue
        
        if normalize:
            std_dev = np.std(raw_signal[i])
            if std_dev != 0:
                processed_signal[j] = (raw_signal[i] - np.mean(raw_signal[i])) / std_dev
            else:
                processed_signal[j] = raw_signal[i] - np.mean(raw_signal[i])
        else:
            processed_signal[j] = raw_signal[i]

    return processed_signal


def filter_signals(raw_signal, freq_min=60, freq_max=200, sampling_frequency=20000):
    """
    Applies a bandpass filter to the signal using SpikeInterface.

    Args:
        raw_signal (np.ndarray): The signal to filter (n_channels, n_samples).
        freq_min (int): The lower bound of the frequency band.
        freq_max (int): The upper bound of the frequency band.
        sampling_frequency (int): The sampling rate of the signal.

    Returns:
        np.ndarray: The filtered signal.
    """
    # SpikeInterface expects the signal in (n_samples, n_channels) format.
    recording = NumpyRecording(raw_signal.T, sampling_frequency=sampling_frequency)
    filtered_recording = bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
    # Transpose back to (n_channels, n_samples).
    filtered_signals = filtered_recording.get_traces().T
    return filtered_signals
