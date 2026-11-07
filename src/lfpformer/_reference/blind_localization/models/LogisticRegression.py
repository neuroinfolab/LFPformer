import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class LogisticRegression(nn.Module):
    """
    A simple Logistic Regression model implemented as a PyTorch Module.
    It consists of a single linear layer followed by a softmax activation.
    """
    def __init__(self, input_size, num_classes):
        """
        Initializes the Logistic Regression model.

        Args:
            input_size (int): The number of input features.
            num_classes (int): The number of output classes.
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes) with class probabilities.
        """
        out = self.linear(x)
        # Apply softmax to get probability distribution over classes.
        out = nn.functional.softmax(out, dim=1)
        return out

    def predict(self, inputs, device):
        """
        Makes predictions for a given set of inputs.

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
            # Get the class with the highest probability.
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
        
        # Standard training loop
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

            # Accumulate loss and predictions for metrics calculation.
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
    return running_loss / len(dataloader), accuracy, f1
