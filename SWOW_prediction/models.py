from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.nn.models import GAT, GCN
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.norm.layer_norm import LayerNorm


class GNNModelSelector:
    """Helper class to select and configure GNN models."""

    @staticmethod
    def get_model(
        model_name: Literal["GAT", "GAT2", "GCN"],
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_heads: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
    ) -> BasicGNN:
        """Factory method to create a GNN model based on specified parameters.

        Args:
            model_name: The type of GNN model to create
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            num_layers: Number of layers in the GNN
            num_heads: Number of attention heads (for GAT)
            out_channels: Number of output features
            dropout: Dropout probability

        Returns:
            Configured GNN model instance

        """
        if model_name == "GAT":
            return GAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=out_channels,
                dropout=dropout,
                v2=False,
                heads=num_heads,
                add_self_loops=False,
            )
        if model_name == "GAT2":
            return GAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=out_channels,
                dropout=dropout,
                v2=True,
                heads=num_heads,
            )
        if model_name == "GCN":
            return GCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=out_channels,
                dropout=dropout,
                heads=num_heads,
                add_self_loops=False,
                jk="max",
                act="tanh",
            )
        raise ValueError(
            f"Unsupported model: {model_name}. Choose from 'GAT', 'GAT2', or 'GCN'",
        )


class BasicPropertyPredictor(nn.Module):
    """Basic module for predicting node properties."""

    def __init__(self, in_channels: int, dropout: float = 0.0):
        """Initialize the property predictor.

        Args:
            in_channels: Number of input features
            dropout: Dropout probability

        """
        super().__init__()

        self.linear1 = nn.Linear(in_channels, in_channels, bias=True)
        self.linear2 = nn.Linear(in_channels, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for node property prediction.

        Args:
            x: Node features tensor

        Returns:
            Predicted property values

        """
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x.squeeze(1)


class ParameterPropertyPredictor(nn.Module):
    """Advanced property predictor that uses both forward and backward embeddings.
    """

    VALID_REDUCTION_METHODS = {
        "mean",
        "sum",
        "add",
        "mult",
        "concat",
        "forward",
        "backward",
        "both",
        "linking",
    }

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        num_layers: int = 2,
        num_heads: int = 5,
        dropout: float = 0.0,
        encoder_model_name: Literal["GAT", "GAT2", "GCN"] = "GCN",
        reduce: str = "mean",
        use_linear_transform: bool = False,
    ):
        """Initialize the parameter property predictor.

        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features (defaults to hidden_channels if None)
            num_layers: Number of layers in the GNN
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout probability
            encoder_model_name: Type of GNN model to use
            reduce: Method to combine forward and backward embeddings
            use_linear_transform: Whether to use additional linear transformations

        """
        super().__init__()

        if reduce not in self.VALID_REDUCTION_METHODS:
            raise ValueError(
                f"Invalid reduction method: {reduce}. Choose from {self.VALID_REDUCTION_METHODS}",
            )

        self.num_nodes = num_nodes
        self.reduce = reduce
        self.use_linear_transform = use_linear_transform

        if out_channels is None:
            out_channels = hidden_channels

        self.encoder_model = GNNModelSelector.get_model(
            encoder_model_name,
            in_channels,
            hidden_channels,
            num_layers,
            num_heads,
            out_channels,
            dropout,
        )
        self.decoder_model = GNNModelSelector.get_model(
            encoder_model_name,
            in_channels,
            hidden_channels,
            num_layers,
            num_heads,
            out_channels,
            dropout,
        )

        self.encoder_norm = LayerNorm(out_channels, mode="node")
        self.decoder_norm = LayerNorm(out_channels, mode="node")

        linear_dim = out_channels * 2 if reduce == "concat" else out_channels
        self.final_linear = nn.Linear(linear_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.emb_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channels)

        if use_linear_transform:
            self.forward_linear = nn.Linear(out_channels, out_channels, bias=False)
            self.backward_linear = nn.Linear(out_channels, out_channels, bias=False)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        edge_attr: Tensor | None = None,
        num_sampled_nodes_per_hop: list[int] | None = None,
        num_sampled_edges_per_hop: list[int] | None = None,
    ) -> Tensor:
        """Forward pass combining both encoder and decoder outputs.

        Args:
            x: Node features tensor
            edge_index: Graph connectivity
            edge_weight: Edge weights
            edge_attr: Edge attributes
            num_sampled_nodes_per_hop: Number of sampled nodes per hop
            num_sampled_edges_per_hop: Number of sampled edges per hop

        Returns:
            Predicted parameter values

        """
        if x.shape[0] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[0]}")

        model_kwargs = {
            "edge_weight": edge_weight,
            "edge_attr": edge_attr,
            "num_sampled_nodes_per_hop": num_sampled_nodes_per_hop,
            "num_sampled_edges_per_hop": num_sampled_edges_per_hop,
        }

        forward_emb = self.encoder_model(x, edge_index, **model_kwargs)
        forward_emb = self.encoder_norm(forward_emb)

        backward_emb = self.decoder_model(x, edge_index, **model_kwargs)
        backward_emb = self.decoder_norm(backward_emb)

        if self.use_linear_transform:
            forward_emb = self.forward_linear(forward_emb)
            backward_emb = self.backward_linear(backward_emb)

        # Combine embeddings based on reduction method
        if self.reduce == "mean":
            node_emb = (forward_emb + backward_emb) / 2
        elif self.reduce in ("sum", "add"):
            node_emb = forward_emb + backward_emb
        elif self.reduce == "mult":
            node_emb = forward_emb * backward_emb
        elif self.reduce == "concat":
            node_emb = torch.cat((forward_emb, backward_emb), dim=1)
        elif self.reduce == "linking":
            similarities = torch.matmul(forward_emb, backward_emb.T)
            node_emb = torch.matmul(similarities, forward_emb)
        elif self.reduce == "forward":
            node_emb = forward_emb
        elif self.reduce == "backward":
            node_emb = backward_emb
        elif self.reduce == "both":
            # Residual connection
            x_transformed = self.relu(self.emb_linear(self.dropout(x)))
            node_emb = x_transformed + forward_emb

        # Final processing and prediction
        if (
            self.reduce != "concat"
        ):  # Skip batch norm for concat to handle different dimensions
            node_emb = self.batch_norm(node_emb)

        output = self.final_linear(node_emb)
        return output.squeeze(1)
