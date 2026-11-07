import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model with one hidden layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the MLP model.

        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim (int): The number of neurons in the hidden layer.
            output_dim (int): The number of output classes.
        """
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output logits.
        """
        return self.model(x)

    def predict(self, inputs, device):
        """
        Makes class predictions for a given set of inputs.

        Args:
            inputs (torch.Tensor): The input data.
            device (str): The device to run the prediction on.

        Returns:
            torch.Tensor: The predicted class labels.
        """
        self.eval()  # Set the model to evaluation mode.
        with torch.no_grad():
            inputs = inputs.to(device).float()
            outputs = self.forward(inputs)
            # Get the index of the max logit as the prediction.
            _, predicted_labels = torch.max(outputs, 1)
        return predicted_labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training for the model.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): The DataLoader providing training data.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimization algorithm.
        device (str): The device to run training on ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the average loss and accuracy for the epoch.
    """
    model.train()  # Set the model to training mode.
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device).float(), labels.to(device).long()
        
        # Standard PyTorch training loop.
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss and calculate accuracy.
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): The DataLoader providing evaluation data.
        criterion (nn.Module): The loss function.
        device (str): The device to run evaluation on.

    Returns:
        tuple: A tuple containing the average loss, accuracy, and weighted F1 score.
    """
    model.eval()  # Set the model to evaluation mode.
    running_loss = 0.0
    correct = 0
    total = 0
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate loss and predictions for metric calculation.
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
    return running_loss / len(dataloader), accuracy, f1

def get_embeddings(model, dataloader, device):
    """
    Extracts the embeddings from the hidden layer of the MLP.

    Args:
        model (MLP): The trained MLP model.
        dataloader (DataLoader): The DataLoader for the input data.
        device (str): The device to run the model on.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The extracted embeddings.
            - np.ndarray: The corresponding labels.
    """
    model.eval() # Set the model to evaluation mode.
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device).float()
            # Pass through the first linear layer and ReLU activation.
            hidden = model.model[0](inputs)
            activated = model.model[1](hidden)
            embeddings.append(activated.cpu().numpy())
            labels.append(label.cpu().numpy())

    return np.concatenate(embeddings), np.concatenate(labels)
