import math

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class BaseCriterion(nn.Module):
    """Abstract base class for loss functions (criteria)."""
    def __init__(self):
        super(BaseCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        """Builds the criterion based on a configuration object."""
        raise NotImplementedError

    def forward(self, model, batch, device):
        """Computes the loss for a given model, batch, and device."""
        raise NotImplementedError
    

class PretrainMaskedCriterion(BaseCriterion):
    """
    Loss function for the masked spectrogram pre-training task.
    
    This criterion calculates a weighted sum of L1 loss on all masked tokens and a 
    content-aware loss that focuses on non-zero (more informative) parts of the signal.
    """
    def __init__(self, alpha):
        """
        Args:
            alpha (float): The weight for the content-aware loss component.
        """
        super(PretrainMaskedCriterion, self).__init__()
        self.alpha = alpha

    def forward(self, output: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor):
        """
        Calculates the loss between the predicted and true spectrograms at masked positions.

        Args:
            output (torch.Tensor): The model's predicted spectrograms.
            truth (torch.Tensor): The ground truth spectrograms.
            mask (torch.Tensor): A boolean mask indicating which parts of the input were masked.

        Returns:
            torch.Tensor: The computed loss value.
        """
        predicted = output.masked_select(mask)
        true_activity = truth.masked_select(mask)
        
        # Standard L1 loss on all masked elements.
        l1 = torch.mean(torch.abs(true_activity - predicted))
        
        # Content-aware loss focusing on non-zero ground truth values.
        non_zero_idxs = torch.abs(true_activity) > 1
        non_zero = torch.mean(torch.abs(true_activity[non_zero_idxs] - predicted[non_zero_idxs]))
        content_aware_loss = self.alpha * non_zero
        
        loss = l1 + content_aware_loss
        return loss


class BaseModel(nn.Module):
    """Abstract base class for models, providing a common interface."""
    def __init__(self):
        super(BaseModel, self).__init__()

    def build_model(self, cfg):
        """Builds the model architecture based on a configuration object."""
        raise NotImplementedError

    def save_model_weights(self, states):
        """Saves the model's state dictionary."""
        states["model"] = self.state_dict() 
        return states

    def load_weights(self, states):
        """Loads weights from a state dictionary."""
        self.load_state_dict(states)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer models.
    Injects information about the relative or absolute position of tokens in the sequence.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer makes `pe` a part of the model's state, but not a parameter to be trained.
        self.register_buffer('pe', pe)

    def forward(self, seq):
        """
        Adds positional encoding to the input sequence.

        Args:
            seq (torch.Tensor): The input sequence, with shape [batch, len, dim].

        Returns:
            tuple: A tuple containing the sequence with positional encoding added, and the positional encoding itself.
        """
        assert len(seq.shape) == 3
        pos_enc = self.pe[:,:seq.size(1),:]
        out = seq + pos_enc
        return out, pos_enc


class TransformerEncoderInput(nn.Module):
    """
    Processes the input spectrograms before they are fed into the Transformer encoder.
    This includes linear projection, adding positional encodings, layer normalization, and dropout.
    """
    def __init__(self, cfg, dropout=0.1):
        super(TransformerEncoderInput, self).__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(in_features=cfg.input_dim, out_features=cfg.hidden_dim)
        self.positional_encoding = PositionalEncoding(self.cfg.hidden_dim)
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_specs):
        """Forward pass for the input processing block."""
        input_specs = self.in_proj(input_specs)
        input_specs, pos_enc = self.positional_encoding(input_specs)
        input_specs = self.layer_norm(input_specs)
        input_specs = self.dropout(input_specs)
        return input_specs, pos_enc


class SpecPredictionHead(nn.Module): 
    """
    The head of the BrainBERT model, used for predicting the original spectrogram from the hidden states.
    This is used during the masked pre-training phase.
    """
    def __init__(self, cfg):
        super(SpecPredictionHead, self).__init__()
        self.hidden_layer = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.act_fn = nn.GELU() if cfg.layer_activation=="gelu" else None
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.output = nn.Linear(cfg.hidden_dim, cfg.input_dim)

    def forward(self, hidden):
        """Projects the Transformer's hidden representation back to the spectrogram space."""
        h = self.hidden_layer(hidden)
        if self.act_fn: h = self.act_fn(h)
        h = self.layer_norm(h)
        h = self.output(h)
        return h


class BrainBERTModel(BaseModel):
    """
    The core BrainBERT model, a Transformer encoder designed for spectrogram data.
    """
    def __init__(self):
        super(BrainBERTModel, self).__init__()

    def forward(self, input_specs: torch.Tensor, src_key_mask=None, rep_from_layer=-1):
        """
        Forward pass of the BrainBERT model.

        Args:
            input_specs (torch.Tensor): The input spectrogram tensor.
            src_key_mask (torch.Tensor, optional): Mask for the source key padding.
            rep_from_layer (int): Not implemented. Intended to extract representation from an intermediate layer.

        Returns:
            tuple: A tuple containing the reconstructed spectrogram, positional encoding, and intermediate representation.
        """
        # Process input (projection, positional encoding, etc.)
        input_specs, pos_enc = self.input_encoding(input_specs)
        # Transformer expects [seq, batch, dim], so we transpose.
        input_specs = input_specs.transpose(0,1)
        
        if rep_from_layer!=-1:
            raise NotImplementedError("Extracting from intermediate layers is not yet implemented.")

        # Pass through the Transformer encoder.
        output_specs: torch.Tensor = self.transformer(input_specs, src_key_padding_mask=src_key_mask)
        # Transpose back to [batch, seq, dim].
        output_specs = output_specs.transpose(0,1)
        intermediate_rep = output_specs
        
        # Predict the original spectrogram from the final hidden state.
        output_specs = self.spec_prediction_head(output_specs)
        return output_specs, pos_enc, intermediate_rep

    def init_weights(self, module):
        """Initializes weights for linear and layer norm layers."""
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def build_model(self, cfg):
        """Constructs the BrainBERT model from a configuration object."""
        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim
        self.input_encoding = TransformerEncoderInput(cfg)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=self.cfg.nhead,
                                                   dim_feedforward=self.cfg.layer_dim_feedforward,
                                                   activation=self.cfg.layer_activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.encoder_num_layers)
        self.spec_prediction_head = SpecPredictionHead(cfg)
        self.apply(self.init_weights)


def train(model, dataloader: DataLoader, optimizer: Optimizer, criterion, device='cpu'):
    """
    Performs one epoch of pre-training for the BrainBERT model.

    Args:
        model: The BrainBERT model to be trained.
        dataloader: DataLoader providing the training data.
        optimizer: The optimization algorithm.
        criterion: The loss function (e.g., PretrainMaskedCriterion).
        device: The device to run the training on ('cpu' or 'cuda').

    Returns:
        float: The average loss over the epoch.
    """
    model.train()
    total_loss = []

    for (spectrograms, augmented_spectrograms), _ in dataloader:
        # The input is the augmented (masked) spectrogram, the truth is the original.
        input_tensor: torch.Tensor = augmented_spectrograms
        truth: torch.Tensor = spectrograms

        # Reshape and transpose to match model's expected input format.
        input_tensor = input_tensor.reshape((input_tensor.shape[0], 500, -1)).transpose(1, 2)
        # Create a mask for the loss function based on zeroed-out (masked) time steps.
        mask = torch.all(input_tensor==0, dim=2).unsqueeze(2)
        truth = truth.reshape((input_tensor.shape[0], 500, -1)).transpose(1, 2)

        input_tensor = input_tensor.to(device)
        mask  = mask.to(device)
        truth = truth.to(device)
        optimizer.zero_grad()

        # Forward pass and loss calculation.
        pred = model(input_tensor)[0]
        loss: torch.Tensor = criterion(pred, truth, mask)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss


def validation(model, dataloader, criterion, device='cpu'):
    """
    Performs one epoch of validation for the pre-trained BrainBERT model.

    Args:
        model: The BrainBERT model to be validated.
        dataloader: DataLoader providing the validation data.
        criterion: The loss function.
        device: The device to run validation on.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    total_loss = []

    with torch.no_grad():
        for (spectrograms, augmented_spectrograms), _ in dataloader:
            input_tensor: torch.Tensor = augmented_spectrograms
            truth: torch.Tensor = spectrograms

            input_tensor = input_tensor.reshape((input_tensor.shape[0], 500, -1)).transpose(1, 2)
            mask = torch.all(input_tensor==0, dim=2).unsqueeze(2)
            truth = truth.reshape((input_tensor.shape[0], 500, -1)).transpose(1, 2)

            input_tensor = input_tensor.to(device)
            mask  = mask.to(device)
            truth = truth.to(device)

            pred = model(input_tensor)[0]
            loss: torch.Tensor = criterion(pred, truth, mask)
            total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    return avg_loss


class BrainBERTWithMLP(nn.Module):
    """
    A model that combines a pre-trained BrainBERT encoder with an MLP head for downstream classification.
    """
    def __init__(self, encoder: BrainBERTModel, input_dim, hidden_dim, output_dim):
        """
        Args:
            encoder (BrainBERTModel): The pre-trained BrainBERT encoder.
            input_dim (int): The input dimension of the MLP (flattened output of the encoder).
            hidden_dim (int): The hidden dimension of the MLP.
            output_dim (int): The output dimension of the MLP (number of classes).
        """
        super(BrainBERTWithMLP, self).__init__()

        self.encoder: BrainBERTModel = encoder
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the combined model.
        
        Returns:
            tuple: A tuple containing the reconstructed input from the encoder and the classification output from the MLP.
        """
        # Get the intermediate representation from the encoder.
        recon_input, _, inter_rep = self.encoder(x)
        # Pass the representation through the MLP for classification.
        output = self.mlp(inter_rep)
        return recon_input, output
    

def freeze_model(model):
    """Sets requires_grad to False for all parameters of a model."""
    for parameter in model.parameters():
        parameter.requires_grad = False


def train_decoder(model: BrainBERTWithMLP, dataloader, optimizer: Optimizer,
                  encoder_criterion, supervised_criterion, device='cpu',
                  mode="separate"):
    """
    Trains the downstream classifier (decoder) with the BrainBERT encoder.

    Args:
        model (BrainBERTWithMLP): The combined encoder-decoder model.
        dataloader: The DataLoader for training data.
        optimizer: The optimizer.
        encoder_criterion: The loss function for the encoder's reconstruction task (used in end-to-end mode).
        supervised_criterion: The loss function for the supervised classification task.
        device: The device to run on.
        mode (str): Training mode. "separate" freezes the encoder and only trains the MLP.
                    "end_to_end" fine-tunes the entire model.

    Returns:
        tuple: A tuple containing the average loss and accuracy for the epoch.
    """
    if mode == "separate":
        # Freeze the encoder weights.
        freeze_model(model.encoder)
    else:
        # Unfreeze the encoder weights for fine-tuning.
        for parameter in model.encoder.parameters():
            parameter.requires_grad = True

    model.train()
    total_loss = []
    total_samples, correct_predictions = 0, 0

    for (spectrograms, augmented_spectrograms), label in dataloader:

        augmented_spectrograms = augmented_spectrograms.reshape((augmented_spectrograms.shape[0], 500, -1)).transpose(1, 2)
        mask = torch.all(augmented_spectrograms==0, dim=2).unsqueeze(2)
        spectrograms = spectrograms.reshape((spectrograms.shape[0], 500, -1)).transpose(1, 2)
        
        augmented_spectrograms = augmented_spectrograms.to(device)
        mask = mask.to(device)
        spectrograms = spectrograms.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        recon_input_i, output_i = model(augmented_spectrograms)

        # Calculate encoder reconstruction loss only if in end-to-end fine-tuning mode.
        encoder_loss = 0
        if mode != "separate":
            encoder_loss = encoder_criterion(recon_input_i, spectrograms, mask)

        # Calculate the supervised classification loss.
        label = label.long()
        supervised_loss = supervised_criterion(output_i, label)
        loss: torch.Tensor = encoder_loss + supervised_loss

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        # Calculate accuracy.
        _, predicted = torch.max(output_i, 1)
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.size(0)

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


def validate_decoder(model: BrainBERTWithMLP, dataloader,
                     supervised_criterion, device='cpu'):
    """
    Validates the downstream classifier.

    Args:
        model (BrainBERTWithMLP): The combined encoder-decoder model.
        dataloader: The DataLoader for validation data.
        supervised_criterion: The loss function for the supervised classification task.
        device: The device to run on.

    Returns:
        tuple: A tuple containing the average loss, accuracy, and weighted F1 score.
    """
    model.eval()
    total_loss = []
    total_samples, correct_predictions = 0, 0
    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for (spectrograms, _), label in dataloader:

            spectrograms = spectrograms.reshape((spectrograms.shape[0], 500, -1)).transpose(1, 2)

            spectrograms = spectrograms.to(device)
            label = label.to(device)
            label = label.long()

            # We only need the classification output for validation.
            _, output = model(spectrograms)
            loss: torch.Tensor = supervised_criterion(output, label)
            total_loss.append(loss.item())

            # Calculate accuracy and F1 score.
            _, predicted = torch.max(output, 1)
            all_true_labels.extend(label.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    avg_loss = sum(total_loss) / len(total_loss)
    accuracy = correct_predictions / total_samples
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
    return avg_loss, accuracy, f1
