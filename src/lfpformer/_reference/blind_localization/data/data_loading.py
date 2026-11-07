import json
import os

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


def load_channel_map(file_path):
    """
    Loads channel coordinate data from an HDF5 file and computes the pairwise
    Euclidean distance matrix between all channels.

    Args:
        file_path (str): The absolute path to the HDF5 file containing channel coordinates.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The channel map with coordinates, sorted lexographically.
            - np.ndarray: The pairwise Euclidean distance matrix between channels.
    """
    filepath = os.path.abspath(file_path)

    # Load x and y coordinates from the HDF5 file.
    with h5py.File(filepath, 'r') as file:
        x_coord = file['xcoords'][:]
        y_coord = file['ycoords'][:]

    # Combine x and y coordinates into a single array.
    channel_map = np.concatenate([x_coord, y_coord], axis=1)
    # Sort channels lexographically by y then x coordinates in batches of 128.
    channel_map = np.concatenate([channel_map[i*128:i*128+128][np.lexsort((channel_map[i*128:i*128+128, 0], channel_map[i*128:i*128+128, 1]))] for i in range(8)], axis=0)

    # Prepare for distance calculation.
    xi = np.insert(channel_map, 1, np.ones(len(channel_map)), axis=1)
    # Calculate the pairwise Euclidean distance between all channels.
    D = cdist(xi, xi, 'euclidean')

    return channel_map, D


def custom_train_test_split(indices, ridx_map, train_size, test_size=0.2, random_state=66, min_samples_per_class=2):
    """
    Splits data into training and testing sets with custom logic to ensure representation.

    This function attempts to satisfy the following conditions in order:
    1. Hold out a fixed percentage of data for each class for testing.
    2. Ensure a minimum number of training samples is available for each class.
    3. Attempt to meet the specified train size for each class with any remaining samples.

    Args:
        indices (np.ndarray): The array of indices to be split.
        ridx_map (np.ndarray): A map of indices to region IDs (classes).
        train_size (float): The desired proportion of the dataset to include in the train split.
        test_size (float): The proportion of the dataset to hold out for the test split for each class.
        random_state (int): The seed used by the random number generator.
        min_samples_per_class (int): The minimum number of samples required for each class in the training set.

    Returns:
        tuple: A tuple containing:
            - list: The list of training indices.
            - list: The list of testing indices.
    """
    regions = np.unique(ridx_map[:, 1])
    np.random.seed(random_state)
    indices_train = []
    indices_test = []

    # Iterate over each unique region (class).
    for r in regions:
        idx_region_r = indices[ridx_map[:, 1] == r]
        n_samples = len(idx_region_r)

        if n_samples < min_samples_per_class:
            raise ValueError(f"Not enough samples for class {r}")
        
        # Shuffle indices to ensure random splits.
        np.random.shuffle(idx_region_r)
        # Assign a fixed percentage of samples to the test set.
        cls_test_indices = idx_region_r[:int(test_size*n_samples)]
        
        # Use the remaining samples for training.
        idx_left = np.setdiff1d(idx_region_r, cls_test_indices)
        # Determine the actual number of training samples to use.
        adjusted_train_size = min(max(min_samples_per_class, int(train_size * n_samples)), n_samples)
        # Randomly select training samples from the remaining pool.
        cls_train_indices = np.random.choice(idx_left, adjusted_train_size, replace=False)

        indices_train.extend(cls_train_indices)
        indices_test.extend(cls_test_indices)

    return indices_train, indices_test


def custom_cv_split(indices, ridx_map, n_splits=5, random_state=66, min_samples_per_class=2):
    """
    Generates indices for cross-validation splits with custom logic.

    Args:
        indices (np.ndarray): The array of indices to be split.
        ridx_map (np.ndarray): A map of indices to region IDs (classes).
        n_splits (int): The number of cross-validation folds.
        random_state (int): The seed used by the random number generator.
        min_samples_per_class (int): The minimum number of samples required for each class.

    Yields:
        tuple: A tuple containing the training and testing indices for each CV fold.
    """
    regions = np.unique(ridx_map[:, 1])
    np.random.seed(random_state)

    # Generate each fold.
    for fold in range(n_splits):
        indices_train = []
        indices_test = []

        # Iterate over each unique region (class).
        for r in regions:
            idx_region_r = indices[ridx_map[:, 1] == r]
            n_samples = len(idx_region_r)

            if n_samples < min_samples_per_class:
                raise ValueError(f"Not enough samples for class {r}")
            
            # Shuffle indices to ensure random splits.
            np.random.shuffle(idx_region_r)
            # Determine the size of each fold.
            fold_size = n_samples // n_splits
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold != n_splits - 1 else n_samples
            
            # Assign the current fold's data to the test set.
            cls_test_indices = idx_region_r[start:end]
            # Use the remaining data for the training set.
            cls_train_indices = np.setdiff1d(idx_region_r, cls_test_indices)

            indices_train.extend(cls_train_indices)
            indices_test.extend(cls_test_indices)

        yield indices_train, indices_test


def load_session_data(json_file_path, session_name):
    """
    Loads session-specific data from a JSON file.

    Args:
        json_file_path (str): The path to the JSON file.
        session_name (str): The name of the session to load data for.

    Returns:
        dict: The data for the specified session.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        return data.get(session_name)
    

def get_train_test_indices(channel_ridx_map, random_state=66, train_size=0.8, test_size=0.2, custom_split=False, min_samples_per_class=2):
    """
    Generates training, validation, and test indices from a channel-region map.

    Channels marked with region ID 5 ("unknown") are always assigned to the test set.

    Args:
        channel_ridx_map (np.ndarray): Map of channel indices to region IDs.
        random_state (int): Seed for the random number generator.
        train_size (float): Proportion of the dataset for the training split.
        test_size (float): Proportion of the dataset for the validation split.
        custom_split (bool): Whether to use the custom splitting logic.
        min_samples_per_class (int): Minimum samples per class for custom splitting.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Channel indices for the training set.
            - np.ndarray: Channel indices for the validation set.
            - np.ndarray: Channel indices for the test set.
            - tuple: Original split indices (train, val, test).
    """
    # Channels with region ID 5 (unknown) are designated for the test set.
    idx_test = np.arange(len(channel_ridx_map))[channel_ridx_map[:, 1] == 5]

    # Create a mask to exclude test indices from the training/validation pool.
    mask = np.isin(np.arange(len(channel_ridx_map)), idx_test)
    masked_channel_region_map = channel_ridx_map[~mask]

    indices = np.arange(len(masked_channel_region_map))

    # Split the remaining data into training and validation sets.
    if custom_split:
        idx_train, idx_val = custom_train_test_split(indices, masked_channel_region_map, train_size, test_size=test_size, random_state=random_state, min_samples_per_class=min_samples_per_class)
    else:
        idx_train, idx_val = train_test_split(indices, test_size=1-train_size, random_state=random_state)

    # Map the split indices back to the original channel indices.
    channel_idx_train = channel_ridx_map[~mask][:, 0][idx_train]
    channel_idx_val = channel_ridx_map[~mask][:, 0][idx_val]
    channel_idx_test = channel_ridx_map[mask][:, 0]
    indices = idx_train, idx_val, idx_test

    return channel_idx_train, channel_idx_val, channel_idx_test, indices


def load_train_test_data(channel_features, D, channel_ridx_map, random_state=66, 
                         train_size=0.8, val_size=0.2, custom_split=False, transferred_channel_idx_train=None, min_samples_per_class=2):
    """
    Loads and splits the data into training, validation, and test sets.

    This function prepares the channel features, distance matrices, and labels for
    each data split based on the provided channel-region map.

    Args:
        channel_features (np.ndarray): The feature matrix for all channels.
        D (np.ndarray): The pairwise distance matrix between all channels.
        channel_ridx_map (np.ndarray): Map of channel indices to region IDs.
        random_state (int): Seed for the random number generator.
        train_size (float): Proportion of the dataset for the training split.
        val_size (float): Proportion of the dataset for the validation split.
        custom_split (bool): Whether to use the custom splitting logic.
        transferred_channel_idx_train (np.ndarray, optional): If provided, these indices are used
            as the reference for distance calculations, for transfer learning scenarios. Defaults to None.
        min_samples_per_class (int): Minimum samples per class for custom splitting.

    Returns:
        tuple: A tuple containing:
            - tuple: Training data (features, distances).
            - tuple: Validation data (features, distances).
            - tuple: Test data (features, distances).
            - np.ndarray: Training labels.
            - np.ndarray: Validation labels.
    """
    # Get the indices for each data split.
    channel_idx_train, channel_idx_val, channel_idx_test, indices = get_train_test_indices(channel_ridx_map, random_state=random_state, train_size=train_size, test_size=val_size, custom_split=custom_split, min_samples_per_class=min_samples_per_class)
    idx_train, idx_val, idx_test = indices
    
    # Create a mask for the test set.
    mask = np.isin(np.arange(len(channel_ridx_map)), idx_test)

    # Group channel indices by split.
    channel_indices = {
        'train': channel_ridx_map[~mask][:, 0][idx_train],
        'val': channel_ridx_map[~mask][:, 0][idx_val],
        'test': channel_ridx_map[mask][:, 0]
    }

    # Create feature sets for each split.
    channel_features_sets = {key: channel_features[indices] for key, indices in channel_indices.items()}

    # Create distance matrix sets for each split.
    # By default, distances are relative to the training set channels.
    distances_sets = {
        'train': D[np.ix_(channel_idx_train, channel_idx_train)],
        'val': D[np.ix_(channel_idx_val, channel_idx_train)],
        'test': D[np.ix_(channel_idx_test, channel_idx_train)]
    }

    # For transfer learning, distances can be relative to a different set of channels.
    if transferred_channel_idx_train is not None:
        distances_sets = {
        'train': D[np.ix_(channel_idx_train, transferred_channel_idx_train)],
        'val': D[np.ix_(channel_idx_val, transferred_channel_idx_train)],
        'test': D[np.ix_(channel_idx_test, transferred_channel_idx_train)]
        }

    # Package data into X (features, distances) and y (labels) sets.
    X_train = (channel_features_sets['train'], distances_sets['train'])
    X_val = (channel_features_sets['val'], distances_sets['val'])
    X_test = (channel_features_sets['test'], distances_sets['test'])

    y_train = channel_ridx_map[~mask][idx_train][:, 1]
    y_val = channel_ridx_map[~mask][idx_val][:, 1]

    return X_train, X_val, X_test, y_train, y_val

def load_results(results_path, field='test_acc'):
    """
    Loads and processes results from a CSV file.

    It separates results for "within" and "across" session experiments.

    Args:
        results_path (str): The path to the CSV results file.
        field (str): The column name in the CSV to extract as the result metric.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of "within" session results.
            - np.ndarray: An array of "across" session results.
    """
    df = pd.read_csv(results_path)

    # Filter dataframe for "within" and "across" experiments.
    within_df = df[df['Name'].str.contains('within')]
    across_df = df[df['Name'].str.contains('across')]

    # Extract and reverse the order of the results.
    within_accs = within_df[field].to_numpy()[::-1]
    across_accs = across_df[field].to_numpy()[::-1]

    return within_accs, across_accs
