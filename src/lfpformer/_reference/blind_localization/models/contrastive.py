import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveEncoder(nn.Module):
    """
    A simple multi-layer perceptron (MLP) encoder for contrastive learning.

    This network takes a flattened input and passes it through a series of
    fully connected layers to produce a lower-dimensional embedding.
    """
    def __init__(self, fc_layer_size, input_size, output_size):
        """
        Initializes the encoder.

        Args:
            fc_layer_size (int): The size of the first hidden layer.
            input_size (int): The dimensionality of the input features.
            output_size (int): The dimensionality of the output embedding.
        """
        super(ContrastiveEncoder, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, fc_layer_size),
            nn.ReLU(),
            nn.Linear(fc_layer_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output embedding.
        """
        x = self.feature_extractor(x)
        return x


class InfoNCELoss(nn.Module):
    """
    Implementation of the InfoNCE (Noise-Contrastive Estimation) loss, a staple in contrastive learning.

    This loss function encourages positive pairs (e.g., two augmented views of the same image)
    to have similar representations while pushing negative pairs (representations from different images)
    apart in the embedding space.
    """
    def __init__(self, temperature=0.5, device='cpu'):
        """
        Initializes the InfoNCE loss module.

        Args:
            temperature (float): A temperature scaling parameter that controls the sharpness
                                 of the distribution of similarities. Lower temperatures
                                 lead to sharper distributions.
            device (str): The device ('cpu' or 'cuda') on which to perform calculations.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, z_i, z_j):
        """
        Calculates the InfoNCE loss for two batches of augmented views.
        Adopted from SimCLR loss: https://github.com/sthalles/SimCLR/blob/master/simclr.py

        Args:
            z_i (torch.Tensor): The first batch of embeddings (e.g., from the first augmented view).
            z_j (torch.Tensor): The second batch of embeddings (e.g., from the second augmented view).

        Returns:
            torch.Tensor: The calculated InfoNCE loss.
        """
        # Concatenate the two batches of embeddings along the batch dimension.
        features = torch.cat([z_i, z_j], dim=0)
        batch_size = len(z_i)

        # Create a large label matrix to identify positive pairs.
        # The diagonal entries of the two blocks correspond to positive pairs.
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)])
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # Normalize the features to lie on the unit hypersphere.
        features = F.normalize(features, dim=1)
        # Calculate the cosine similarity matrix.
        similarity_matrix = torch.matmul(features, features.T)

        # Discard the diagonal elements (self-similarity).
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Select the positive samples (similarities between z_i and z_j).
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # Select the negative samples (all other similarities).
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        # Debugging checks for numerical stability.
        if torch.isnan(positives).any() or torch.isnan(negatives).any():
            print("NaN detected in positives or negatives")
        if torch.isinf(positives).any() or torch.isinf(negatives).any():
            print("Inf detected in positives or negatives")

        # Create the logits for the cross-entropy loss.
        # The first column contains the positive similarities.
        logits = torch.cat([positives, negatives], dim=1)
        
        # More debugging checks.
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN or Inf detected in logits")
            
        # The labels for cross-entropy are all zeros because the positive sample is always in the first column.
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # Scale logits by the temperature.
        logits = logits / self.temperature
        # Calculate the final cross-entropy loss.
        loss = self.criterion(logits, labels)
        return loss
