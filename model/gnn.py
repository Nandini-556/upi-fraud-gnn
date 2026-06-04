"""
Fraud GNN Model — GraphSAGE + GAT architecture.

Layer 1 & 2  : SAGEConv  — fast mean-aggregation, great for large graphs
Layer 3      : GATConv   — attention weights, explainable per-neighbor scores
Classifier   : Linear    — node-level fraud probability

Wrapped with to_hetero() so it handles (user, txn) node types automatically.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, SAGEConv, to_hetero


class _FraudGNNBase(torch.nn.Module):
    """
    Homogeneous GNN backbone.
    to_hetero() clones weights for each node/edge type.
    """

    def __init__(self, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.hidden  = hidden
        self.dropout = dropout

        # SAGEConv uses lazy init (in_channels=-1) — infers input dim on first forward
        self.conv1 = SAGEConv(-1, hidden, normalize=True)
        self.conv2 = SAGEConv(hidden, hidden, normalize=True)
        self.conv3 = GATConv(hidden, hidden, heads=4, concat=False,
                             dropout=0.1, add_self_loops=False)

        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.bn2 = torch.nn.BatchNorm1d(hidden)

        self.classifier = torch.nn.Linear(hidden, 2)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Hop 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Hop 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Hop 3 — GAT with attention
        x = self.conv3(x, edge_index)
        x = F.elu(x)

        return self.classifier(x)


def build_model(metadata: tuple, hidden: int = 64) -> torch.nn.Module:
    """
    Returns a heterogeneous GNN ready for (user, txn) graph.

    metadata = (node_types, edge_types) from data.metadata()
    """
    base  = _FraudGNNBase(hidden=hidden)
    model = to_hetero(base, metadata=metadata, aggr="sum")
    return model


class FraudPredictor(torch.nn.Module):
    """
    Thin wrapper around the hetero GNN.
    Exposes a predict() method that returns fraud probabilities.
    """

    def __init__(self, metadata: tuple, hidden: int = 64):
        super().__init__()
        self.gnn = build_model(metadata, hidden)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        return self.gnn(x_dict, edge_index_dict)

    @torch.no_grad()
    def predict(self, x_dict: dict, edge_index_dict: dict) -> Tensor:
        """Returns fraud probability for every txn node (float tensor)."""
        self.eval()
        logits = self.forward(x_dict, edge_index_dict)
        return logits["txn"].softmax(dim=-1)[:, 1]   # P(fraud)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Model saved → {path}")

    def load(self, path: str, device: str = "cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f"Model loaded ← {path}")
