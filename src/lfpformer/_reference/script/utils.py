# This file contains utility functions used across the region decoding project.
# It includes functions for calculating performance metrics, signal processing tasks like
# spectrogram generation and power spectral density calculation, and file handling.

import os
import librosa
import pickle
import numpy as np

from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from collections import Counter
from scipy.signal import welch
from scipy.signal import spectrogram
import torch

def calculate_balanced_accuracy(train_real_labels_all, train_pred_labels_all, test_real_labels_all,
                                test_pred_labels_all, sessions_list, title):
    """
    Calculates and prints a suite of performance metrics for both training and testing sets.

    For each session, it computes:
    - Balanced Accuracy: Accuracy that accounts for class imbalance.
    - Standard Accuracy: Raw percentage of correct predictions.
    - Macro F1-Score: Unweighted mean of the F1-scores for each class.
    - Micro F1-Score: Global F1-score calculated from the total true positives, etc.
    - Chance Level Accuracy: Accuracy of a baseline classifier that always predicts the most frequent class.

    Args:
        train_real_labels_all (list): A list of arrays, where each array contains the true labels for a training session.
        train_pred_labels_all (list): A list of arrays with the predicted labels for a training session.
        test_real_labels_all (list): A list of arrays with the true labels for a testing session.
        test_pred_labels_all (list): A list of arrays with the predicted labels for a testing session.
        sessions_list (list): A list of session identifiers, corresponding to the data lists.
        title (str): A title for the print output to identify the model or experiment.
    """
    # --- Calculate and print metrics for the training set ---
    print(f"\n--- Metrics for {title} ---")
    print("Training Set Metrics:")
    for i in range(len(train_real_labels_all)):
        train_real_labels = train_real_labels_all[i]
        train_pred_labels = train_pred_labels_all[i]

        # Calculate various performance scores.
        bal_train_acc = balanced_accuracy_score(train_real_labels, train_pred_labels)
        train_acc = accuracy_score(train_real_labels, train_pred_labels)
        bal_train_f1_score = f1_score(train_real_labels, train_pred_labels, average='macro')
        train_f1_score = f1_score(train_real_labels, train_pred_labels, average='micro')

        # Calculate the chance level accuracy as a baseline.
        label_counts = Counter(train_real_labels)
        most_frequent_label_count = label_counts.most_common(1)[0][1]
        chance_level_accuracy = most_frequent_label_count / len(train_real_labels)

        # Print the results for the current training session.
        print(f"Title: {title}, Train Session: {sessions_list[i]}, CA: {chance_level_accuracy:.5f}, "
              f"BTA: {bal_train_acc:.5f}, BTF1: {bal_train_f1_score:.5f}, "
              f"TA: {train_acc:.5f}, TF1: {train_f1_score:.5f}")

    # --- Calculate and print metrics for the test set ---
    print("\nTest Set Metrics:")
    for i in range(len(test_real_labels_all)):
        test_real_labels = test_real_labels_all[i]
        test_pred_labels = test_pred_labels_all[i]

        # Calculate various performance scores.
        bal_test_acc = balanced_accuracy_score(test_real_labels, test_pred_labels)
        test_acc = accuracy_score(test_real_labels, test_pred_labels)
        bal_test_f1_score = f1_score(test_real_labels, test_pred_labels, average='macro')
        test_f1_score = f1_score(test_real_labels, test_pred_labels, average='micro')

        # Calculate the chance level accuracy as a baseline.
        label_counts = Counter(test_real_labels)
        most_frequent_label_count = label_counts.most_common(1)[0][1]
        chance_level_accuracy = most_frequent_label_count / len(test_real_labels)

        # Print the results for the current test session.
        print(f"Title: {title}, Test Session: {sessions_list[i]}, CA: {chance_level_accuracy:.5f}, "
              f"BTA: {bal_test_acc:.5f}, BTF1: {bal_test_f1_score:.5f}, "
              f"TA: {test_acc:.5f}, TF1: {test_f1_score:.5f}")


def zscore(a, axis):
    """
    Computes the z-score normalization of a NumPy array or a PyTorch tensor along a specified axis.
    Z-score is calculated as (value - mean) / std.

    Args:
        a (np.ndarray or torch.Tensor): The input data to be normalized.
        axis (int): The axis along which to compute the mean and standard deviation.

    Returns:
        The z-score normalized array or tensor.
    """
    # Check if the input is a NumPy array or a PyTorch tensor and calculate mean/std accordingly.
    if isinstance(a, np.ndarray):
        mean = a.mean(axis=axis, keepdims=True)
        std = a.std(axis=axis, ddof=0, keepdims=True)
    elif isinstance(a, torch.Tensor):
        mean = a.mean(dim=axis, keepdim=True)
        std = a.std(dim=axis, unbiased=False, keepdim=True)

    std[std == 0] = 1.0  # Prevent division by zero where standard deviation is zero.
    z = (a - mean) / std
    return z


def compute_spectrogram(x, fs, num_freq_bins=500, time_bins=18, normalizing='db', return_onesided=True, **kwargs):
    """
    Computes the spectrogram of a signal using scipy.signal.spectrogram.

    Args:
        x (np.ndarray): The input time series signal.
        fs (int): The sampling frequency of the signal.
        num_freq_bins (int): The number of frequency bins to keep in the final output.
        time_bins (int): The desired number of time bins in the final output.
        normalizing (str): The type of normalization to apply ('db' for decibels or 'zscore').
        return_onesided (bool): If True, returns a one-sided spectrogram for real inputs.
        **kwargs: Additional arguments passed to scipy.signal.spectrogram.

    Returns:
        np.ndarray: The computed and transposed spectrogram (time x frequency).
    """
    # Calculate nperseg and noverlap to achieve the desired number of time bins.
    nperseg = 2 * (num_freq_bins - 1)
    noverlap = int(nperseg - len(x) / time_bins)

    # Compute the spectrogram using SciPy's implementation.
    f, t, Zxx = spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap, return_onesided=return_onesided, **kwargs)

    # Truncate to the desired number of frequency bins.
    Zxx = Zxx[:num_freq_bins, :]
    f = f[:num_freq_bins]

    # Take the magnitude of the complex-valued spectrogram.
    Zxx = np.abs(Zxx)

    # Apply the specified normalization.
    if normalizing == "zscore":
        Zxx = zscore(Zxx, axis=-1)
        # If normalization results in a constant value, replace with ones.
        if (Zxx.std() == 0).any():
            Zxx = np.ones_like(Zxx)
    elif normalizing == "db":
        # Convert to decibels (log scale), adding a small epsilon to avoid log(0).
        Zxx = np.log(Zxx + 1e-10)

    # Replace any potential NaNs with zero.
    if np.isnan(Zxx).any():
        Zxx = np.nan_to_num(Zxx, nan=0.0)

    # The following lines are commented out but can be used to visualize the spectrogram.
    # plt.figure(figsize=(10, 6))
    # plt.pcolormesh(t, f, Zxx, shading='gouraud', cmap='inferno')
    # plt.colorbar(label='Power (dB)')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (s)')
    # plt.title('Spectrogram')
    # plt.savefig("spectrogram.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # Transpose the result to have time as the first dimension.
    return np.transpose(Zxx)


def save_if_better(output_path, val_acc, train_real_labels, train_pred_labels, test_real_labels, test_pred_labels):
    """
    Saves model results to a pickle file only if the current validation accuracy is better
    than a previously saved version.

    Args:
        output_path (str): The path to the output pickle file.
        val_acc (float): The validation accuracy of the current model.
        train_real_labels (np.ndarray): True labels for the training set.
        train_pred_labels (np.ndarray): Predicted labels for the training set.
        test_real_labels (np.ndarray): True labels for the test set.
        test_pred_labels (np.ndarray): Predicted labels for the test set.
    """
    # Check if a result file already exists.
    if os.path.exists(output_path):
        # If it exists, load the previously saved validation accuracy.
        with open(output_path, 'rb') as f:
            saved_data = pickle.load(f)
            saved_val_acc = saved_data['decoder_val_acc']

        # Compare the current validation accuracy with the saved one.
        if val_acc > saved_val_acc:
            # If the current model is better, overwrite the file with the new results.
            print(f"New validation accuracy ({val_acc}) is higher than saved accuracy ({saved_val_acc})"
                  f", overwriting the file.")
            with open(output_path, 'wb') as f:
                pickle.dump({'decoder_val_acc': val_acc, 'train_real_labels': train_real_labels,
                             'train_pred_labels': train_pred_labels, 'test_real_labels': test_real_labels,
                             'test_pred_labels': test_pred_labels}, f)
        else:
            print(f"Saved validation accuracy ({saved_val_acc}) is higher, keeping the old file.")
    else:
        # If no file exists, create a new one with the current results.
        with open(output_path, 'wb') as f:
            print(f"No existing file found. Saving new file with validation accuracy: {val_acc}.")
            pickle.dump({'decoder_val_acc': val_acc, 'train_real_labels': train_real_labels,
                         'train_pred_labels': train_pred_labels, 'test_real_labels': test_real_labels,
                         'test_pred_labels': test_pred_labels}, f)

def compute_spectrogram_librosa(x, fs, num_freq_bins=500, time_bins=16, axis=0, **kwargs):
    """
    Computes a log-power spectrogram of a signal using the librosa library.

    Args:
        x (np.ndarray): The input time series signal.
        fs (int): The sampling frequency of the signal (not used by librosa.stft but kept for consistency).
        num_freq_bins (int): The number of frequency bins to keep in the final output.
        time_bins (int): The desired number of time bins in the final output.
        axis (int): The axis of the input data that holds the time series.
        **kwargs: Additional arguments (not used in this implementation).

    Returns:
        np.ndarray: The computed log-power spectrogram.
    """
    nperseg = 2048  # Define the segment length for the STFT.
    total_signal_length = x.shape[0]
    # Calculate the hop length to achieve the desired number of time bins.
    hop_length = total_signal_length // (time_bins - 1)
    
    # Compute the Short-Time Fourier Transform (STFT).
    spectrogram = librosa.stft(x, n_fft=nperseg, hop_length=hop_length,
                                   win_length=nperseg)
    # Convert the complex-valued STFT to a power spectrogram.
    spectrogram = np.abs(spectrogram) ** 2
    epsilon = 1e-10  # Add a small constant to avoid log(0).
    # Convert the power spectrogram to a log scale (decibels).
    spectrogram = 10 * np.log10(spectrogram + epsilon)
    # Keep only the specified number of frequency bins.
    spectrogram = spectrogram[:500]

    return spectrogram


def calculate_psd(data, sampling_rate, freq_range=None):
    """
    Calculates the Power Spectral Density (PSD) of a signal using Welch's method.

    Args:
        data (np.ndarray): The input time series signal.
        sampling_rate (int): The sampling frequency of the signal.
        freq_range (tuple, optional): A tuple (min_freq, max_freq) to filter the frequencies.
                                     If max_freq is -1, no upper bound is applied.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The frequencies.
            - np.ndarray: The power spectral density.
    """
    # Compute PSD using Welch's method.
    freqs, Pxx = welch(data, fs=sampling_rate, nperseg=1024)

    # If a frequency range is specified, filter the results.
    if freq_range is not None:
        min_freq, max_freq = freq_range
        if max_freq == -1:
            # No upper frequency limit.
            mask = (freqs >= min_freq)
        else:
            # Frequency range with both lower and upper limits.
            mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs = freqs[mask]
        Pxx = Pxx[mask]

    return freqs, Pxx


def calculate_power(data, sampling_rate, freq_range=None):
    """
    Calculates the total power within a specified frequency band by integrating the PSD.

    Args:
        data (np.ndarray): The input time series signal.
        sampling_rate (int): The sampling frequency of the signal.
        freq_range (tuple, optional): The frequency range over which to calculate the power.

    Returns:
        tuple: A tuple containing:
            - float: The total power in the band.
            - float: The Root Mean Square (RMS) of the PSD values in the band.
    """
    # First, calculate the Power Spectral Density.
    freqs, Pxx = calculate_psd(data, sampling_rate, freq_range)

    # Calculate the total power by integrating the PSD over the frequencies (trapezoidal rule).
    power = np.trapz(Pxx, freqs)
    # Calculate the RMS of the PSD values themselves.
    rms = calculate_rms(Pxx)
    return power, rms


def calculate_rms(data):
    """
    Calculates the Root Mean Square (RMS) of a signal.

    Args:
        data (np.ndarray): The input signal.

    Returns:
        float: The RMS value of the signal.
    """
    return np.sqrt(np.mean(data ** 2))
