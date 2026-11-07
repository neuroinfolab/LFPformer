import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphEncoder(nn.Module):
    """
    A Graph Convolutional Network (GCN) based encoder for contrastive learning on graphs.
    This implementation is inspired by the GRACE (Deep GRAph Contrastive rEpresentation learning) model.
    It also includes the contrastive loss function from the paper.
    Reference: https://github.com/CRIPAC-DIG/GRACE
    """
    def __init__(self, in_channels, out_channels, k, tau):
        """
        Initializes the GraphEncoder.

        Args:
            in_channels (int): Number of input features per node.
            out_channels (int): Number of output features per node.
            k (int): The number of GCN layers in the encoder.
            tau (float): The temperature parameter for the contrastive loss.
        """
        super(GraphEncoder, self).__init__()
        self.k = k
        self.tau = tau
        self.activation = nn.ReLU()
        
        # Build the GCN layers.
        layers = []
        layers.append(GCNConv(in_channels, out_channels * 2))
        for _ in range(1, k - 1):
            layers.append(GCNConv(out_channels * 2, out_channels * 2))
        layers.append(GCNConv(out_channels * 2, out_channels))
        
        # Using nn.ModuleList because nn.Sequential cannot handle the `edge_index` argument in GCNConv's forward pass.
        self.encoder = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Performs the forward pass through the GCN encoder.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The graph connectivity in COO format.

        Returns:
            torch.Tensor: The node embeddings after passing through the encoder.
        """
        # Manually iterate through the GCN layers.
        for i in range(self.k):
            x = self.activation(self.encoder[i](x, edge_index))
        return x
    
    def cosine_sim(self, z1, z2):
        """
        Calculates the cosine similarity between two sets of embeddings.

        Args:
            z1 (torch.Tensor): The first set of embeddings (N, D).
            z2 (torch.Tensor): The second set of embeddings (N, D).

        Returns:
            torch.Tensor: The pairwise cosine similarity matrix (N, N).
        """
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        """
        Calculates the semi-loss for one view against another, as defined in the GRACE paper.
        This is the core component of the contrastive loss for a single direction (z1 -> z2).

        Args:
            z1 (torch.Tensor): Embeddings from the first view.
            z2 (torch.Tensor): Embeddings from the second view.

        Returns:
            torch.Tensor: A tensor of loss values for each node in the view.
        """
        # Temperature-scaled exponential function.
        f = lambda x: torch.exp(x / self.tau)
        
        # Similarities between nodes within the same view (intra-view).
        intra_view_sim = f(self.cosine_sim(z1, z1))
        # Similarities between nodes across different views (inter-view).
        inter_view_sim = f(self.cosine_sim(z1, z2))

        # The loss for each node `i` is -log(positive_pair_sim / (all_pairs_sim)).
        # Positive pair is the node `i` in z1 and z2.
        # Negative pairs are all other pairs, both intra-view and inter-view.
        return -torch.log(
            inter_view_sim.diag()
            / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag()))

    def batched_semi_loss(self, z1, z2, batch_size):
        """
        Calculates the semi-loss in batches to reduce memory consumption.
        This is useful for very large graphs where the full N x N similarity matrix doesn't fit in memory.

        Args:
            z1 (torch.Tensor): Embeddings from the first view.
            z2 (torch.Tensor): Embeddings from the second view.
            batch_size (int): The size of the batches to process.

        Returns:
            torch.Tensor: A tensor of loss values for all nodes.
        """
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            # Select a batch of nodes.
            mask = indices[i * batch_size:(i + 1) * batch_size]

            # Calculate similarity of the batch against all nodes in both views.
            intra_view_sim = f(self.cosine_sim(z1[mask], z1))  # Shape: [B, N]
            inter_view_sim = f(self.cosine_sim(z1[mask], z2))  # Shape: [B, N]

            # Calculate loss for the current batch.
            batch_loss = -torch.log(
                inter_view_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (intra_view_sim.sum(1) + inter_view_sim.sum(1)
                   - intra_view_sim[:, i * batch_size:(i + 1) * batch_size].diag()))
            losses.append(batch_loss)

        return torch.cat(losses)

    def loss(self, z1, z2, mean=True, batch_size=0):
        """
        Calculates the final GRACE contrastive loss.
        It's the average of the semi-loss in both directions (z1 -> z2 and z2 -> z1).

        Args:
            z1 (torch.Tensor): Embeddings from the first view.
            z2 (torch.Tensor): Embeddings from the second view.
            mean (bool): If True, returns the mean loss; otherwise, returns the sum.
            batch_size (int): If > 0, uses the batched loss calculation.

        Returns:
            torch.Tensor: The final scalar loss value.
        """
        if batch_size == 0:
            l1 = self.semi_loss(z1, z2)
            l2 = self.semi_loss(z2, z1)
        else:
            l1 = self.batched_semi_loss(z1, z2, batch_size)
            l2 = self.batched_semi_loss(z2, z1, batch_size)

        # The total loss is the average of the two directional semi-losses.
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    

class CombinedEncoder(nn.Module):
    """
    A model that combines two different encoders and fuses their representations.
    One encoder is graph-based (encoder_a), and the other is feature-based (encoder_x).
    The concatenated embeddings are passed through a final MLP.
    """
    def __init__(self, encoder_x, encoder_a, input_dim, hidden_dim, output_dim, k):
        """
        Initializes the CombinedEncoder.

        Args:
            encoder_x: The feature-based encoder.
            encoder_a: The graph-based encoder.
            input_dim (int): The input dimension to the final MLP (concatenated embedding size).
            hidden_dim (int): The hidden dimension of the final MLP.
            output_dim (int): The output dimension of the final MLP.
            k (int): The number of layers in the final MLP.
        """
        super(CombinedEncoder, self).__init__()
        
        self.encoder_x = encoder_x
        self.encoder_a = encoder_a
        
        # Build the final MLP for combining features.
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(1, k - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Sigmoid())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.ReLU())

        self.encoder_combined = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        """
        Forward pass for the combined encoder.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The graph connectivity.

        Returns:
            tuple: A tuple containing the concatenated intermediate embeddings (z) and the final output.
        """
        # Get embeddings from both encoders.
        z1 = self.encoder_a(x, edge_index) # Graph-based embedding
        z2 = self.encoder_x(x)             # Feature-based embedding
        
        # Concatenate the embeddings.
        z = torch.cat([z1, z2], dim=1)

        # Pass the combined embedding through the final MLP.
        output = self.encoder_combined(z)
        return z, output
