from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from blind_localization.data.datasets import *
from blind_localization.models.contrastive import *


def build_single_session_dataloader(channel_features_all, channel_labels_all, channel_trials_all,
                                    config, Dataset, sessions, session_idx=0, vit=False):
    """
    Builds training, validation, and test DataLoaders for a single session.

    The data is split based on trial indices, ensuring that data from the same trial
    does not appear in both the training and testing sets.

    Args:
        channel_features_all (dict): A dictionary mapping session names to feature arrays.
        channel_labels_all (dict): A dictionary mapping session names to label arrays.
        channel_trials_all (dict): A dictionary mapping session names to trial index arrays.
        config (dict): A configuration dictionary containing parameters like batch size.
        Dataset (torch.utils.data.Dataset): The Dataset class to use.
        sessions (list): A list of all session names.
        session_idx (int): The index of the session to use for building the dataloaders.
        vit (bool): Flag indicating if the data should be preprocessed for a Vision Transformer.

    Returns:
        tuple: A tuple containing the train, validation, and test DataLoaders.
    """
    features = channel_features_all.get(sessions[session_idx])
    labels = channel_labels_all.get(sessions[session_idx])
    trials = channel_trials_all.get(sessions[session_idx])
    features = np.array(features)

    # Split trials into training, validation, and test sets.
    trial_length = 60 # Assuming a fixed number of trials
    train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
    train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2

    # Get the corresponding data indices for each trial split.
    train_idx = [idx for idx, val in enumerate(trials) if val in train_tr_idx]
    val_idx = [idx for idx, val in enumerate(trials) if val in val_tr_idx]
    test_idx = [idx for idx, val in enumerate(trials) if val in test_tr_idx]

    # Create data splits.
    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    # Create Dataset objects.
    train_dataset = Dataset(np.array(X_train), y_train, spectrogram_size=500,
                            time_bins=config["time_bins"], library=config["library"], vit=vit)
    val_dataset = Dataset(np.array(X_val), y_val, spectrogram_size=500,
                          time_bins=config["time_bins"], library=config["library"], vit=vit)
    test_dataset = Dataset(np.array(X_test), y_test, spectrogram_size=500,
                           time_bins=config["time_bins"], library=config["library"], vit=vit)

    # Create DataLoader objects.
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def build_multi_session_dataloader(channel_features_all, channel_labels_all, channel_trials_all,
                                   sessions, config, Dataset, test_session_idx=0, vit=False):
    """
    Builds DataLoaders for a multi-session (cross-session) setup.

    One session is held out for testing, and a subset of the remaining sessions are used for training and validation.

    Args:
        channel_features_all (dict): Dictionary mapping session names to features.
        channel_labels_all (dict): Dictionary mapping session names to labels.
        channel_trials_all (dict): Dictionary mapping session names to trials.
        sessions (list): A list of all session names.
        config (dict): Configuration dictionary.
        Dataset (torch.utils.data.Dataset): The Dataset class to use.
        test_session_idx (int): The index of the session to hold out for testing.
        vit (bool): Flag for Vision Transformer preprocessing.

    Returns:
        tuple: A tuple containing the train, validation, and test DataLoaders.
    """
    # Prepare the test set from the designated test session.
    X_test = np.array(channel_features_all.get(sessions[test_session_idx]))
    y_test = channel_labels_all.get(sessions[test_session_idx])
    trials = channel_trials_all.get(sessions[test_session_idx])

    # Split trials within the test session to create the final test set.
    trial_length = 60
    train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
    train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.25, random_state=42)
    test_idx = [idx for idx, val in enumerate(trials) if val in test_tr_idx]
    X_test = X_test[test_idx]
    y_test = np.array(y_test)[test_idx]

    # Prepare training and validation sets from other sessions.
    X_train, y_train, X_val, y_val = [], [], [], []
    sessions_except_test = [sess for sess in sessions if sess != sessions[test_session_idx]]
    # Use a random sample of 4 other sessions for training/validation.
    sessions_to_use = random.sample(sessions_except_test, 4)
    for sess in sessions_to_use:
        features = np.array(channel_features_all.get(sess))
        labels = channel_labels_all.get(sess)
        trial_idx = channel_trials_all.get(sess)

        # Split based on the same trial indices used for the test session.
        idx_train = [idx for idx, val in enumerate(trial_idx) if val in train_tr_idx]
        idx_val = [idx for idx, val in enumerate(trial_idx) if val in val_tr_idx]

        X_train.extend(features[idx_train])
        y_train.extend(np.array(labels)[idx_train])

        X_val.extend(features[idx_val])
        y_val.extend(np.array(labels)[idx_val])
    
    print(f"Train shape: {np.array(X_train).shape}, Test shape: {np.array(X_test).shape}")

    # Create Dataset and DataLoader objects.
    train_dataset = Dataset(np.array(X_train), y_train, spectrogram_size=500,
                            time_bins=config["time_bins"], library=config["library"], vit=vit)
    val_dataset = Dataset(np.array(X_val), y_val, spectrogram_size=500,
                          time_bins=config["time_bins"], library=config["library"], vit=vit)
    test_dataset = Dataset(np.array(X_test), y_test, spectrogram_size=500,
                           time_bins=config["time_bins"], library=config["library"], vit=vit)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

def build_multi_session_dataloader_new(channel_features_all, channel_labels_all, channel_trials_all,channel_chans_all,
                                   sessions, config, Dataset, test_session_idx=0, vit=False):
    """
    An updated function to build DataLoaders for a multi-session setup, including metadata tracking.

    This version provides a more explicit split of sessions for training and validation
    and returns metadata strings for each data point.

    Args:
        channel_features_all (dict): Dictionary of features per session.
        channel_labels_all (dict): Dictionary of labels per session.
        channel_trials_all (dict): Dictionary of trial IDs per session.
        channel_chans_all (dict): Dictionary of channel IDs per session.
        sessions (list): List of all session names.
        config (dict): Configuration dictionary.
        Dataset (torch.utils.data.Dataset): The Dataset class.
        test_session_idx (int): Index of the test session.
        vit (bool): Flag for Vision Transformer preprocessing.

    Returns:
        tuple: A tuple of (train_loader, val_loader, test_loader, train_meta, val_meta, test_meta).
    """
    # Prepare test set from the designated test session.
    X_test = channel_features_all.get(sessions[test_session_idx])
    y_test = channel_labels_all.get(sessions[test_session_idx])
    trials = channel_trials_all.get(sessions[test_session_idx])
    chans = channel_chans_all.get(sessions[test_session_idx])

    # Split trials into train/val/test portions.
    trial_length = 60
    train_tr_idx, test_tr_idx = train_test_split(range(trial_length), test_size=0.2, random_state=42)
    train_tr_idx, val_tr_idx = train_test_split(train_tr_idx, test_size=0.25, random_state=42)
    
    # Select data for the test set based on test trial indices.
    test_indices = [idx for idx, val in enumerate(trials) if val in test_tr_idx]
    X_test = [X_test[i] for i in test_indices]
    y_test = [y_test[i] for i in test_indices]
    trials_test = [trials[i] for i in test_indices]
    chans_test = [chans[i] for i in test_indices]
    test_metadata = [f"{sessions[test_session_idx]}_{trials_test[i]}_{chans_test[i]}" for i in range(len(trials_test))]
    assert len(X_test) == len(y_test) == len(trials_test) == len(chans_test)

    # Prepare training and validation sets from the remaining sessions.
    train_metadata, val_metadata = [], []
    X_train, y_train, X_val, y_val = [], [], [], []
    available_sessions = [sess for sess in sessions if sess != sessions[test_session_idx]]
    random.seed(42)
    train_sessions = available_sessions.copy()
    val_sessions = [train_sessions[-1]] # Use the last of the training sessions for validation.

    # Aggregate training data.
    for sess in train_sessions:
        features = channel_features_all.get(sess)
        labels = channel_labels_all.get(sess)
        trials = channel_trials_all.get(sess)
        chans = channel_chans_all.get(sess)
        idx_train = [idx for idx, val in enumerate(trials) if val in train_tr_idx]
        X_train.extend([features[i] for i in idx_train])
        y_train.extend([labels[i] for i in idx_train])
        trials_train = [trials[i] for i in idx_train]
        chans_train = [chans[i] for i in idx_train]
        train_metadata.extend([f"{sess}_{trials_train[i]}_{chans_train[i]}" for i in range(len(trials_train))])

    # Aggregate validation data.
    for sess in val_sessions:
        features = channel_features_all.get(sess)
        labels = channel_labels_all.get(sess)
        trials = channel_trials_all.get(sess)
        chans = channel_chans_all.get(sess)
        idx_val = [idx for idx, val in enumerate(trials) if val in val_tr_idx]
        X_val.extend([features[i] for i in idx_val])
        y_val.extend([labels[i] for i in idx_val])
        trials_val = [trials[i] for i in idx_val]
        chans_val = [chans[i] for i in idx_val]
        val_metadata.extend([f"{sess}_{trials_val[i]}_{chans_val[i]}" for i in range(len(trials_val))])
    
    print(f"Training on {train_sessions}, Validate on {val_sessions}, Test on {sessions[test_session_idx]}")
    print(f"Training shape: {np.array(X_train).shape}, Validation shape: {np.array(X_val).shape}, Test shape: {np.array(X_test).shape}")

    # Create Dataset and DataLoader objects.
    train_dataset = Dataset(np.array(X_train), y_train, spectrogram_size=500, time_bins=config["time_bins"], library=config["library"], vit=vit)
    val_dataset = Dataset(np.array(X_val), y_val, spectrogram_size=500, time_bins=config["time_bins"], library=config["library"], vit=vit)
    test_dataset = Dataset(np.array(X_test), y_test, spectrogram_size=500, time_bins=config["time_bins"], library=config["library"], vit=vit)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, train_metadata, val_metadata, test_metadata


def extract_features(model, dataloader, device='cpu', feature_flag=True):
    """
    Extracts features (embeddings) from a model for a given dataset.

    Args:
        model (nn.Module): The encoder model.
        dataloader (DataLoader): The DataLoader for the data.
        device (str): The device to run the model on.
        feature_flag (bool): If True, pass data through the model. If False, use the raw data as features.

    Returns:
        tuple: A tuple of (features_array, labels_array).
    """
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            signal = data[0].to(device)
            label = label.to(device)
            labels.append(label)
            # Pass signal through the model to get embeddings, or use the signal directly.
            feature = model(signal) if feature_flag else signal
            features.append(feature)

    features = torch.cat(features, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    return features, labels


def train_clf_with_embeddings(model, dataloader, device, subset_size=2000):
    """
    Trains a linear SVM classifier on embeddings extracted from a model.

    This is a common way to evaluate the quality of self-supervised representations.

    Args:
        model (nn.Module): The feature encoder.
        dataloader (DataLoader): DataLoader for the training data.
        device (str): The device to run on.
        subset_size (int, optional): The number of samples to use for training the SVM. Defaults to 2000.

    Returns:
        sklearn.svm.SVC: The trained SVM classifier.
    """
    # Extract features from the encoder.
    X_embed, y_embed = extract_features(model, dataloader, device)
    # Optionally, use a subset of the data to speed up training.
    if subset_size and subset_size < X_embed.shape[0]:
        random_indices = np.random.choice(X_embed.shape[0], subset_size, replace=False)
        X_embed, y_embed = X_embed[random_indices], y_embed[random_indices]
    
    # Train the SVM classifier.
    clf = svm.SVC(C=10, gamma='scale')
    clf.fit(X_embed, y_embed)
    return clf


def eval_clf_with_embeddings(model, dataloader, device, clf):
    """
    Evaluates a trained SVM classifier on a new set of embeddings.

    Args:
        model (nn.Module): The feature encoder.
        dataloader (DataLoader): DataLoader for the evaluation data.
        device (str): The device to run on.
        clf (sklearn.svm.SVC): The trained SVM classifier.

    Returns:
        tuple: A tuple containing the predicted labels and the accuracy score.
    """
    # Extract features for the evaluation set.
    X_embed, y_embed = extract_features(model, dataloader, device)
    # Get predictions from the classifier.
    y_pred = clf.predict(X_embed)
    return y_pred, accuracy_score(y_pred, y_embed)


def train(model, dataloader, optimizer, criterion, device='cpu'):
    """
    Performs one epoch of contrastive training.

    Args:
        model (nn.Module): The contrastive encoder model.
        dataloader (DataLoader): The training DataLoader.
        optimizer (Optimizer): The optimizer.
        criterion (nn.Module): The contrastive loss function (e.g., InfoNCELoss).
        device (str): The device to train on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = []

    for data, label in dataloader:
        spectrograms, augmented_spectrograms = data
        spectrograms = spectrograms.to(device)
        augmented_spectrograms = augmented_spectrograms.to(device)
        optimizer.zero_grad()

        # Get embeddings for both augmented views.
        z_i = model(spectrograms)
        z_j = model(augmented_spectrograms)
        
        # Calculate the contrastive loss.
        loss: torch.Tensor = criterion(z_i, z_j)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    
    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss


def validation(model, dataloader, criterion, device='cpu'):
    """
    Performs one epoch of contrastive validation.

    Args:
        model (nn.Module): The contrastive encoder model.
        dataloader (DataLoader): The validation DataLoader.
        criterion (nn.Module): The contrastive loss function.
        device (str): The device to run on.

    Returns:
        float: The average validation loss for the epoch.
    """
    model.eval()
    total_loss = []

    with torch.no_grad():
        for data, label in dataloader:
            spectrograms, augmented_spectrograms = data
            spectrograms = spectrograms.to(device)
            augmented_spectrograms = augmented_spectrograms.to(device)

            # Get embeddings for both augmented views.
            z_i = model(spectrograms)
            z_j = model(augmented_spectrograms)
            
            # Calculate the contrastive loss.
            loss = criterion(z_i, z_j)
            total_loss.append(loss.item())
    
    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss

def set_return_session_trial(train_dataloader, val_dataloader, test_dataloader, flag):
    """
    Utility function to set the `return_session_trial` flag in dataset objects within DataLoaders.

    Args:
        train_dataloader (DataLoader): The training DataLoader.
        val_dataloader (DataLoader): The validation DataLoader.
        test_dataloader (DataLoader): The test DataLoader.
        flag (bool): The boolean value to set the flag to.
    """
    if hasattr(train_dataloader.dataset, 'return_session_trial'):
        train_dataloader.dataset.return_session_trial = flag
    if hasattr(val_dataloader.dataset, 'return_session_trial'):
        val_dataloader.dataset.return_session_trial = flag
    if hasattr(test_dataloader.dataset, 'return_session_trial'):
        test_dataloader.dataset.return_session_trial = flag
