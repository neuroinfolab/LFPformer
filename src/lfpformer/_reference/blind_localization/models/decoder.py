import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score


class ContrastiveLearningWithMLP(nn.Module):
    """
    A model that combines a (potentially pre-trained) encoder with a Multi-Layer Perceptron (MLP) head.
    This is typically used for downstream classification tasks after a self-supervised pre-training phase.
    """
    def __init__(self, encoder, input_dim, hidden_dim, output_dim):
        """
        Initializes the model.

        Args:
            encoder (nn.Module): The feature encoder. Can be a pre-trained model or nn.Identity().
            input_dim (int): The dimension of the features coming from the encoder.
            hidden_dim (int): The dimension of the MLP's hidden layer.
            output_dim (int): The number of output classes.
        """
        super(ContrastiveLearningWithMLP, self).__init__()

        self.encoder = encoder if encoder is not None else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the encoder and then the MLP.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The latent representation (embedding) from the encoder.
                - torch.Tensor: The final classification logits from the MLP.
        """
        z = self.encoder(x)
        output = self.mlp(z)
        return z, output


class ContrastiveLearningWithLR(nn.Module):
    """
    A model that combines an encoder with a simple Logistic Regression (LR) head (a single linear layer).
    This serves as a simpler alternative to the MLP head for linear evaluation protocols.
    """
    def __init__(self, encoder, input_dim, output_dim):
        """
        Initializes the model.

        Args:
            encoder (nn.Module): The feature encoder.
            input_dim (int): The dimension of the features from the encoder.
            output_dim (int): The number of output classes.
        """
        super(ContrastiveLearningWithLR, self).__init__()

        self.encoder = encoder if encoder is not None else nn.Identity()
        self.lr = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the encoder and then the linear layer.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple: A tuple containing the latent representation and the classification logits.
        """
        z = self.encoder(x)
        output = self.lr(z)
        return z, output


def freeze_model(model):
    """Utility function to freeze all parameters of a given model."""
    for parameter in model.parameters():
        parameter.requires_grad = False


def train_decoder(model, dataloader, optimizer, contrastive_criterion, supervised_criterion, device='cpu',
                  mode="separate", lambda_c=1):
    """
    Trains a combined encoder-decoder model for one epoch.

    Supports two modes:
    - "separate": Freezes the encoder and trains only the decoder (classification head).
    - "end_to_end": Fine-tunes both the encoder and decoder jointly.

    Args:
        model (nn.Module): The combined model (e.g., ContrastiveLearningWithMLP).
        dataloader (DataLoader): The DataLoader for training data.
        optimizer (Optimizer): The optimizer.
        contrastive_criterion (nn.Module): The loss function for the contrastive task (used in end-to-end mode).
        supervised_criterion (nn.Module): The loss function for the supervised classification task.
        device (str): The device to run training on.
        mode (str): The training mode ("separate" or "end_to_end").
        lambda_c (float): The weight for the supervised loss component.

    Returns:
        dict: A dictionary of training metrics (loss, accuracy, etc.).
    """
    if mode == "separate":
        freeze_model(model.encoder)
    else:
        # Ensure encoder parameters are trainable for fine-tuning.
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True

    model.train()
    total_loss = []
    total_samples, correct_predictions = 0, 0
    all_true_labels = []
    all_predicted_labels = []

    for data, label in dataloader:
        spectrograms, augmented_spectrograms = data
        spectrograms = spectrograms.to(device)
        augmented_spectrograms = augmented_spectrograms.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        # Forward pass for both original and augmented data.
        z_i, output_i = model(spectrograms)
        z_j, _ = model(augmented_spectrograms)

        # Calculate contrastive loss only if in end-to-end mode.
        contrastive_loss = 0
        if mode != "separate":
            contrastive_loss = contrastive_criterion(z_i, z_j)

        # Calculate supervised classification loss.
        label = label.long()
        supervised_loss = supervised_criterion(output_i, label)
        # Combine the losses.
        loss = contrastive_loss + lambda_c * supervised_loss

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        # Track metrics for reporting.
        _, predicted = torch.max(output_i, 1)
        all_true_labels.extend(label.cpu().numpy())
        all_predicted_labels.extend(predicted.cpu().numpy())
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.size(0)

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples
    balanced_accuracy = balanced_accuracy_score(all_true_labels, all_predicted_labels)
    
    metrics = {"loss": avg_loss, "balanced_accuracy": balanced_accuracy, "accuracy": accuracy,
               "real_labels": all_true_labels, "predicted_labels": all_predicted_labels}

    return metrics


def validate_decoder(model, dataloader, supervised_criterion, device='cpu'):
    """
    Validates the combined encoder-decoder model for one epoch.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): The DataLoader for validation data.
        supervised_criterion (nn.Module): The supervised loss function.
        device (str): The device to run validation on.

    Returns:
        dict: A dictionary of validation metrics (loss, accuracy, F1 scores, etc.).
    """
    model.eval()
    total_loss = []
    total_samples, correct_predictions = 0, 0
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_logits = []
    all_embeddings = []

    with torch.no_grad():
        for data, label in dataloader:
            spectrograms, _ = data
            spectrograms = spectrograms.to(device)
            label = label.to(device)
            label = label.long()

            # Forward pass to get embeddings and logits.
            z, output = model(spectrograms)
            loss = supervised_criterion(output, label)

            # Accumulate metrics.
            total_loss.append(loss.item())
            _, predicted = torch.max(output, 1)
            all_true_labels.extend(label.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            all_predicted_logits.extend(output.cpu().numpy())
            all_embeddings.extend(z.cpu().numpy())

            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples

    # Compile a comprehensive dictionary of metrics.
    metrics = {"loss": avg_loss,
               "accuracy": accuracy,
               "balanced_accuracy": balanced_accuracy_score(np.array(all_true_labels), np.array(all_predicted_labels)),
               "f1": f1_score(np.array(all_true_labels), np.array(all_predicted_labels), average='micro'),
               "weighted_f1": f1_score(np.array(all_true_labels), np.array(all_predicted_labels), average='weighted'),
               "macro_f1": f1_score(np.array(all_true_labels), np.array(all_predicted_labels), average='macro'),
               "real_labels": all_true_labels,
               "predicted_labels": all_predicted_labels,
               "logits": all_predicted_logits,
               "embeddings": all_embeddings}
    return metrics


class GraphDecoder(nn.Module):
    """
    A decoder model for graph-based data, combining a graph encoder with an MLP head.
    """
    def __init__(self, encoder, input_dim, hidden_dim, output_dim):
        """
        Initializes the graph decoder.

        Args:
            encoder (nn.Module): The graph encoder (e.g., a GCN, GAT).
            input_dim (int): The dimension of the node features from the encoder.
            hidden_dim (int): The dimension of the MLP's hidden layer.
            output_dim (int): The number of output classes.
        """
        super(GraphDecoder, self).__init__()

        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index):
        """
        Forward pass for the graph decoder.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The graph connectivity in COO format.

        Returns:
            tuple: A tuple containing the node embeddings and the final classification logits.
        """
        z = self.encoder(x, edge_index)
        output = self.mlp(z)
        return z, output
