import numba
numba.config.DISABLE_JIT = True
import librosa
import scipy
import torch
import torchaudio
import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from scipy.signal import spectrogram


def compute_trial(raw_signal, sr, ts=2160, t_start=2160, dT=0.5):
    """
    Extracts a trial of a specific duration from a raw signal.

    Args:
        raw_signal (np.ndarray): The raw input signal.
        sr (int): The sampling rate of the signal.
        ts (float): The timestamp of the trial's start.
        t_start (float): The reference start time for the trial extraction.
        dT (float): The duration of the trial in seconds.

    Returns:
        np.ndarray: The extracted trial signal.
    """
    ts_start, ts_end = int((ts - t_start) * sr), int((ts - t_start + dT) * sr)
    return raw_signal[:, ts_start:ts_end]


class SpectrogramDataset(Dataset):
    """
    A PyTorch Dataset for spectrogram data, designed for contrastive learning.

    Positive pairs are created from spectrograms of the same channel in neighboring trials.
    """
    def __init__(self, spectrograms, labels, window_size=1):
        """
        Initializes the dataset.

        Args:
            spectrograms (np.ndarray): A numpy array of spectrograms with shape (n_trials, n_channels, n_features).
            labels (np.ndarray): An array of labels for each channel.
            window_size (int): The number of neighboring trials to consider for positive pairs.
        """
        self.spectrograms = spectrograms.astype('float32')
        self.labels = labels
        self.window_size = window_size
        self.n_trials, self.n_channels, self.n_features = spectrograms.shape

    def __len__(self):
        """Returns the total number of samples in the dataset (trials * channels)."""
        return self.n_trials * self.n_channels

    def __getitem__(self, index):
        """
        Retrieves a data sample and its positive pair.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the (data, augmented_data) pair and the corresponding label.
        """
        trial_idx = index // self.n_channels
        channel_idx = index % self.n_channels

        data = self.spectrograms[trial_idx][channel_idx]
        label = self.labels[channel_idx]

        # Select a neighboring trial from the same channel to be a positive pair.
        neighbor_trial_indices = list(range(max(0, trial_idx - self.window_size), trial_idx)) + \
                                 list(range(trial_idx + 1, min(self.n_trials, trial_idx + self.window_size + 1)))
        neighbor_trial_idx = random.choice(neighbor_trial_indices)
        augmented_data = self.spectrograms[neighbor_trial_idx][channel_idx]

        return (data, augmented_data), label


class AvgSpectrogramDataset(Dataset):
    """
    A PyTorch Dataset for spectrograms averaged across trials.

    Positive pairs are created from different channels that share the same label.
    """
    def __init__(self, spectrograms, labels, window_size=1):
        """
        Initializes the dataset.

        Args:
            spectrograms (np.ndarray): A numpy array of spectrograms with shape (n_trials, n_channels, n_features).
            labels (np.ndarray): An array of labels for each channel.
            window_size (int): This parameter is not used in this class but kept for compatibility.
        """
        self.spectrograms = spectrograms.astype('float32')
        self.labels = labels
        self.window_size = window_size
        self.n_trials, self.n_channels, self.n_features = spectrograms.shape
        # Compute the average spectrogram across all trials.
        self.avg_spectrograms = np.mean(self.spectrograms, axis=0)

    def __len__(self):
        """Returns the total number of channels."""
        return self.n_channels

    def __getitem__(self, index):
        """
        Retrieves a data sample and its positive pair.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the (data, augmented_data) pair and the corresponding label.
        """
        data = self.avg_spectrograms[index]
        label = self.labels[index]

        # Select another channel of the same class to be a positive pair.
        same_class_channels_idx = np.arange(self.n_channels)[self.labels == label]
        same_class_channels_idx = np.delete(same_class_channels_idx, np.where(same_class_channels_idx == index))

        if len(same_class_channels_idx) > 0:
            selected_idx = random.choice(same_class_channels_idx)
            augmented_data = self.avg_spectrograms[selected_idx]
        else:
            # If no other channel of the same class exists, use the original data as the pair.
            augmented_data = data

        return (data, augmented_data), label

class OriginalDataset(Dataset):
    """
    A PyTorch Dataset that computes spectrograms from raw signals.

    This class handles the transformation of raw time-series data into spectrograms
    and applies augmentations like temporal masking.
    """
    def __init__(self, raw_signal, labels, sampling_rate=20000, n_fft=2048, n_freq_bins = 500, n_time_bins=16, transform=None, session_labels=None):
        """
        Initializes the dataset.

        Args:
            raw_signal (np.ndarray): The raw input signal.
            labels (np.ndarray): An array of labels for each channel.
            sampling_rate (int): The sampling rate of the raw signal.
            n_fft (int): The number of FFT components.
            n_freq_bins (int): The number of frequency bins to keep in the spectrogram.
            n_time_bins (int): The number of time bins in the spectrogram.
            transform (callable, optional): Optional transform to be applied on a sample.
            session_labels (np.ndarray, optional): Labels for each session, if applicable.
        """
        self.raw_signal = torch.tensor(raw_signal, dtype=torch.float32)
        self.labels = torch.tensor(labels)
        self.n_time_bins = n_time_bins
        self.n_trials, self.n_channels, self.n_features = raw_signal.shape
        self.n_freq_bins, self.n_fft = n_freq_bins, n_fft

        self.hop_length = self.n_features // (self.n_time_bins - 1)
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, power=2.0)
        self.transform = transform
        self.session_labels = session_labels
        self.return_session_trial = True if session_labels is not None else False

        # Precompute spectrograms and augmented spectrograms to save time.
        self.spectrograms, self.augmented_spectrograms = self._compute_spectrogram(self.raw_signal)

    def _compute_spectrogram(self, raw_signal):
        """Computes spectrograms and their augmented versions for the entire dataset."""
        spectrograms = torch.zeros((self.n_trials, self.n_channels, self.n_freq_bins*self.n_time_bins))
        augmented_spectrograms = torch.zeros((self.n_trials, self.n_channels, self.n_freq_bins*self.n_time_bins))

        for trial_idx in range(self.n_trials):
            for channel_idx in range(self.n_channels):
                data = raw_signal[trial_idx][channel_idx]
                spectrogram = 10 * torch.log10(self.spectrogram_transform(data)[:self.n_freq_bins])
                spectrograms[trial_idx][channel_idx] = spectrogram.flatten()
                augmented_spectrogram = self.temporal_masking(spectrogram)
                augmented_spectrograms[trial_idx][channel_idx] = augmented_spectrogram.flatten()

        return spectrograms, augmented_spectrograms

    def temporal_masking(self, spectrogram, percent_masked=0.2, mask_size=1):
        """Applies temporal masking to a spectrogram by setting random time steps to zero."""
        n_masks = int(spectrogram.size(1) * percent_masked)

        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(1) - mask_size, (1,)).item()
            spectrogram[:, start_idx:start_idx+mask_size] = 0

        return spectrogram

    def frequency_masking(self, spectrogram, percent_masked=0.05, mask_size=5):
        """Applies frequency masking to a spectrogram by setting random frequency bands to zero."""
        n_masks = int(spectrogram.size(0) * percent_masked)

        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(0) - mask_size, (1,)).item()
            spectrogram[start_idx:start_idx+mask_size, :] = 0

        return spectrogram

    def __len__(self):
        """Returns the total number of samples in the dataset (trials * channels)."""
        return self.n_trials * self.n_channels

    def __getitem__(self, index):
        """
        Retrieves a data sample and its augmented version.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the (spectrogram, augmented_spectrogram) pair, the label, and optionally the session/trial info.
        """
        trial_idx = index // self.n_channels
        channel_idx = index % self.n_channels

        spectrograms = self.spectrograms[trial_idx][channel_idx]
        augmented_spectrograms = self.augmented_spectrograms[trial_idx][channel_idx]

        label = self.labels[channel_idx]

        if self.return_session_trial:
            session_label = self.session_labels[trial_idx]
            return (spectrograms, augmented_spectrograms), label, (session_label, trial_idx)

        return (spectrograms, augmented_spectrograms), label

class RawDataset(Dataset):
    """
    A PyTorch Dataset for creating spectrograms from raw signals, with support for multiple libraries.

    This class is optimized for performance by pre-computing spectrograms and allows for different
    spectrogram generation backends (PyTorch, Librosa, SciPy).
    """
    def __init__(self, raw_signal, labels, nperseg=2048, spectrogram_size=500, transform=None, library='pytorch',
                 time_bins=None, vit=False):
        """
        Initializes the dataset.

        Args:
            raw_signal (np.ndarray): The raw input signal.
            labels (np.ndarray): An array of labels for each channel.
            nperseg (int): The length of each segment for the STFT.
            spectrogram_size (int): The number of frequency bins to keep.
            transform (callable, optional): Optional transform to be applied on a sample.
            library (str): The library to use for spectrogram computation ('pytorch', 'librosa', or 'scipy').
            time_bins (int, optional): The desired number of time bins in the output spectrogram.
            vit (bool): If True, preprocesses the spectrograms for a Vision Transformer (ViT) model.
        """
        self.raw_signal = torch.tensor(raw_signal, dtype=torch.float32)
        self.labels = torch.tensor(labels)
        self.library = library
        self.sample_rate = 2500  # NOTE: This should be adjusted based on the actual sampling rate of the data.
        self.n_channels, self.n_features = raw_signal.shape

        # Calculate hop_length based on the desired number of time_bins.
        if time_bins is not None:
            total_signal_length = raw_signal.shape[-1]
            if library == 'scipy':
                time_bins = 2 * time_bins + 2
            self.time_bins = time_bins
            hop_length = total_signal_length // (time_bins - 1)
        else:
            hop_length = nperseg // 2

        self.nperseg = nperseg
        self.time_bins = time_bins
        self.hop_length = self.n_features // (self.time_bins - 1)
        self.spectrogram_size = spectrogram_size
        self.transform = transform

        # Precompute spectrograms and their augmented versions.
        self.spectrograms, self.augmented_spectrograms = self._compute_spectrogram(self.raw_signal,
                                                                                   spectrogram_size, self.time_bins)
        # Preprocess for Vision Transformer if specified.
        if vit:
            self.spectrograms = self.vit_preprocess(self.spectrograms)
            self.augmented_spectrograms = self.vit_preprocess(self.augmented_spectrograms)

    @numba.jit
    def _compute_spectrogram(self, raw_signal, spectrogram_size, timebins):
        """Computes spectrograms and their augmented versions for all channels."""
        spectrograms = torch.zeros((self.n_channels, spectrogram_size * timebins))
        augmented_spectrograms = torch.zeros((self.n_channels, spectrogram_size * timebins))

        for channel_idx in range(self.n_channels):
            data = raw_signal[channel_idx]
            if self.library is None or self.library == 'pytorch':
                spectrogram = self._pytorch_spectrogram(data)
            elif self.library == 'librosa':
                spectrogram = self._librosa_spectrogram(data)
            elif self.library == 'scipy':
                spectrogram = self._scipy_spectrogram(data)
            spectrogram = spectrogram[:spectrogram_size]
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
            spectrograms[channel_idx] = spectrogram.flatten()

            augmented_spectrogram = self.temporal_masking(spectrogram)
            augmented_spectrograms[channel_idx] = augmented_spectrogram.flatten()

        return spectrograms, augmented_spectrograms

    def _pytorch_spectrogram(self, data):
        """Computes a spectrogram using PyTorch's torchaudio."""
        spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.nperseg, win_length=self.nperseg, hop_length=self.hop_length, power=2.0
        )
        return 10 * torch.log10(spectrogram_transform(data))

    def _librosa_spectrogram(self, data):
        """Computes a spectrogram using the Librosa library."""
        spectrogram = librosa.stft(data.numpy(), n_fft=self.nperseg, hop_length=self.hop_length,
                                   win_length=self.nperseg)
        spectrogram = np.abs(spectrogram) ** 2
        epsilon = 1e-10
        spectrogram = 10 * np.log10(spectrogram + epsilon)
        return spectrogram

    def _scipy_spectrogram(self, data):
        """Computes a spectrogram using SciPy."""
        frequencies, times, spectrogram = scipy.signal.spectrogram(data.numpy(), nperseg=self.nperseg,
                                                                   noverlap=self.nperseg - self.hop_length)
        return torch.tensor(spectrogram, dtype=torch.float32)

    def vit_preprocess(self, data):
        """Reshapes and resizes spectrogram data for use with a Vision Transformer."""
        reshaped_data = data.reshape(self.n_channels, self.spectrogram_size, self.time_bins)
        reshaped_data = torch.tensor(reshaped_data, dtype=torch.float32)

        # Interpolate to the size expected by the ViT model.
        reshaped_data = F.interpolate(reshaped_data.unsqueeze(1), size=(128, 1024), mode="bilinear", align_corners=False).squeeze(1)
        return reshaped_data

    def temporal_masking(self, spectrogram, percent_masked=0.25, mask_size=1):
        """Applies temporal masking to a spectrogram."""
        n_masks = int(spectrogram.size(1) * percent_masked)

        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(1), (1,)).item()
            spectrogram[:, start_idx:start_idx + mask_size] = 0

        return spectrogram

    def frequency_masking(self, spectrogram, percent_masked=0.05, mask_size=5):
        """Applies frequency masking to a spectrogram."""
        n_masks = int(spectrogram.size(0) * percent_masked)

        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(0) - mask_size, (1,)).item()
            spectrogram[start_idx:start_idx + mask_size, :] = 0

        return spectrogram

    def __len__(self):
        """Returns the total number of channels."""
        return self.n_channels

    def __getitem__(self, index):
        """
        Retrieves a data sample and its augmented version.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the (spectrogram, augmented_spectrogram) pair and the label.
        """
        channel_idx = index % self.n_channels

        spectrograms = self.spectrograms[channel_idx]
        augmented_spectrograms = self.augmented_spectrograms[channel_idx]

        label = self.labels[channel_idx]

        return (spectrograms, augmented_spectrograms), label

    def visualize_spectrograms(self, spectrograms, augmented_spectrograms, shape):
        """Visualizes and saves spectrograms for debugging purposes."""
        def save_spectrogram(spectrogram_data, suffix):
            spectrogram = spectrogram_data[0].reshape((shape[0], shape[1]))
            frequencies = np.arange(0, shape[0])

            time_steps = np.linspace(0, 3, shape[1])

            plt.figure(figsize=(6, 6))
            plt.pcolormesh(time_steps, frequencies, spectrogram, shading='gouraud')
            plt.ylim(0, shape[0])
            plt.colorbar(label='Intensity [dB]')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plt.title(f'{suffix} Spectrogram - Normal Scale')
            plt.tight_layout()
            plt.savefig(f'../results/{suffix}_spectrogram_normal.png')

        save_spectrogram(spectrograms, 'normal')
        save_spectrogram(augmented_spectrograms, 'augmented')

    def visualize_mel_spectrograms(self, raw_signal):
        """Visualizes and saves Mel spectrograms for debugging purposes."""
        def save_mel_spectrogram(raw_signal, suffix, sampling_rate=None):
            raw_signal = raw_signal[0].numpy()
            if sampling_rate is None:
                sampling_rate = self.sample_rate

            mel_spec = librosa.feature.melspectrogram(y=raw_signal, sr=sampling_rate, n_fft=self.nperseg,
                                                      hop_length=self.hop_length, n_mels=128, fmax=500, fmin=0)

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            time_steps = np.linspace(0, 3, 16)
            frequencies = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=500)

            plt.figure(figsize=(6, 6))
            plt.pcolormesh(time_steps, frequencies, mel_spec_db, shading='gouraud')
            plt.colorbar(label='Intensity [dB]')
            plt.yticks(frequencies[::10], frequencies[::10].astype(int))
            plt.title(f'{suffix} Mel Spectrogram')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plt.tight_layout()
            plt.savefig(f'../results/{suffix}_mel_spectrogram.png')

        save_mel_spectrogram(raw_signal, 'normal')


class ImageDataset(Dataset):
    """
    A PyTorch Dataset that provides 2D spectrogram images.

    Similar to RawDataset, but outputs spectrograms as 2D tensors (images) 
    instead of flattened 1D arrays. It also uses a more flexible augmentation system.
    """

    def __init__(self, raw_signal, labels, sampling_rate=20000, n_fft=2048, n_freq_bins=500, n_time_bins=16,
                 transform=None, session_labels=None, augmentation_type=None, aug_params=None):
        """
        Initializes the dataset.

        Args:
            raw_signal (np.ndarray): The raw input signal.
            labels (np.ndarray): An array of labels for each channel.
            sampling_rate (int): The sampling rate of the raw signal.
            n_fft (int): The number of FFT components.
            n_freq_bins (int): The number of frequency bins to keep in the spectrogram.
            n_time_bins (int): The number of time bins in the spectrogram.
            transform (callable, optional): Optional transform to be applied on a sample.
            session_labels (np.ndarray, optional): Labels for each session, if applicable.
            augmentation_type (str, optional): The type of augmentation to apply.
            aug_params (dict, optional): Parameters for the augmentation.
        """
        self.raw_signal = torch.tensor(raw_signal, dtype=torch.float32)
        self.labels = torch.tensor(labels)
        self.n_time_bins = n_time_bins
        self.n_trials, self.n_channels, self.n_features = raw_signal.shape
        self.n_freq_bins, self.n_fft = n_freq_bins, n_fft

        self.hop_length = self.n_features // (self.n_time_bins - 1)
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.n_fft,
                                                                       hop_length=self.hop_length, power=2.0)
        self.transform = transform
        self.session_labels = session_labels
        self.return_session_trial = True if session_labels is not None else False

        # Precompute spectrograms.
        self.spectrograms = self._compute_spectrogram(self.raw_signal)
        self.augmentation = Augmentation(augmentation_type, **(aug_params or {}))

    def _compute_spectrogram(self, raw_signal):
        """Computes 2D spectrograms for the entire dataset."""
        spectrograms = torch.zeros((self.n_trials, self.n_channels, self.n_freq_bins, self.n_time_bins))

        for trial_idx in range(self.n_trials):
            for channel_idx in range(self.n_channels):
                data = raw_signal[trial_idx][channel_idx]
                spectrogram = 10 * torch.log10(self.spectrogram_transform(data)[:self.n_freq_bins])
                spectrograms[trial_idx][channel_idx] = spectrogram

        return spectrograms

    def __len__(self):
        """Returns the total number of samples in the dataset (trials * channels)."""
        return self.n_trials * self.n_channels

    def get_augmentation(self, trial_idx, channel_idx):
        """Applies the configured augmentation to a spectrogram."""
        return self.augmentation.apply(
            self.spectrograms[trial_idx][channel_idx],
            trial_idx=trial_idx,
            channel_idx=channel_idx,
            dataset=self
        )

    def __getitem__(self, index):
        """
        Retrieves a data sample and its augmented version as 2D images.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the (spectrogram, augmented_spectrogram) pair, the label, and optionally the session/trial info.
        """
        trial_idx = index // self.n_channels
        channel_idx = index % self.n_channels

        spectrograms = self.spectrograms[trial_idx][channel_idx].unsqueeze(0)
        augmented_spectrograms = self.get_augmentation(trial_idx, channel_idx).unsqueeze(0)
        label = self.labels[channel_idx]

        if self.return_session_trial:
            session_label = self.session_labels[trial_idx]
            return (spectrograms, augmented_spectrograms), label, (session_label, trial_idx)

        return (spectrograms, augmented_spectrograms), label


class Augmentation:
    """
    A factory class for applying various data augmentation techniques to spectrograms.
    """
    def __init__(self, augmentation_type=None, **kwargs):
        """
        Initializes the Augmentation handler.

        Args:
            augmentation_type (str, optional): The type of augmentation to apply. 
                Options: "temporal_masking", "frequency_masking", "temporal_neighbors", "spatial_neighbors".
            **kwargs: Additional parameters for the chosen augmentation method.
        """
        self.augmentation_type = augmentation_type
        self.params = kwargs

    def apply(self, spectrogram, trial_idx=None, channel_idx=None, dataset=None):
        """
        Applies the specified augmentation to a spectrogram.

        Args:
            spectrogram (torch.Tensor): The input spectrogram.
            trial_idx (int, optional): The trial index, required for neighbor-based augmentations.
            channel_idx (int, optional): The channel index, required for neighbor-based augmentations.
            dataset (Dataset, optional): The dataset object, required for neighbor-based augmentations.

        Returns:
            torch.Tensor: The augmented spectrogram.
        """
        if self.augmentation_type == "temporal_masking":
            return self.temporal_masking(spectrogram, **self.params)
        elif self.augmentation_type == "frequency_masking":
            return self.frequency_masking(spectrogram, **self.params)
        elif self.augmentation_type == "temporal_neighbors":
            return self.temporal_neighbors(trial_idx, channel_idx, dataset)
        elif self.augmentation_type == "spatial_neighbors":
            return self.spatial_neighbors(trial_idx, channel_idx, dataset)
        else:
            # If no augmentation type is specified, return the original spectrogram.
            return spectrogram

    @staticmethod
    def temporal_masking(spectrogram, percent_masked=0.2, mask_size=1):
        """Applies temporal masking by zeroing out random time steps."""
        n_masks = int(spectrogram.size(1) * percent_masked)
        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(1) - mask_size, (1,)).item()
            spectrogram[:, start_idx:start_idx + mask_size] = 0
        return spectrogram

    @staticmethod
    def frequency_masking(spectrogram, percent_masked=0.05, mask_size=5):
        """Applies frequency masking by zeroing out random frequency bands."""
        n_masks = int(spectrogram.size(0) * percent_masked)
        for _ in range(n_masks):
            start_idx = torch.randint(0, spectrogram.size(0) - mask_size, (1,)).item()
            spectrogram[start_idx:start_idx + mask_size, :] = 0
        return spectrogram

    def temporal_neighbors(self, trial_idx, channel_idx, dataset):
        """Uses the spectrogram from the next trial of the same channel as an augmentation."""
        neighbor_trial_idx = (trial_idx + 1) % len(dataset.spectrograms)
        return dataset.spectrograms[neighbor_trial_idx][channel_idx]

    def spatial_neighbors(self, trial_idx, channel_idx, dataset):
        """Uses the spectrogram from the next channel of the same trial as an augmentation."""
        neighbor_channel_idx = (channel_idx + 1) % dataset.n_channels
        return dataset.spectrograms[trial_idx][neighbor_channel_idx]
