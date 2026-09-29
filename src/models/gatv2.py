import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Tuple, Optional
from ..config import ModelConfig

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        
        if weights is not None:
            loss = loss * weights
            
        return loss.mean() if self.reduction == "mean" else loss


class GATv2DiabetesClassifier(nn.Module):
    """
    Graph Attention Network v2 (GATv2) for diabetes risk prediction.
    Supports both standard single-relational k-NN graphs (edge_dim=None, simpler & higher performance)
    and multi-relational graphs with edge embeddings for ablation studies.
    """
    def __init__(self, in_features: int, config: ModelConfig = None, edge_dim: Optional[int] = None):
        super().__init__()
        cfg = config or ModelConfig()
        hidden_dim = cfg.hidden_dim
        heads = cfg.heads
        dropout = cfg.dropout
        self.edge_dim = edge_dim

        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        if edge_dim is not None and edge_dim > 0:
            self.edge_embedding = nn.Embedding(num_embeddings=2, embedding_dim=edge_dim)
            self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
            self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        else:
            self.edge_embedding = None
            self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h = self.input_proj(x)
        h = F.elu(self.norm1(h))
        h = self.dropout(h)

        if self.edge_embedding is not None and edge_type is not None:
            edge_attr = self.edge_embedding(edge_type)
        else:
            edge_attr = None

        if return_attention:
            if edge_attr is not None:
                h_att1, att_weights1 = self.gat1(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            else:
                h_att1, att_weights1 = self.gat1(h, edge_index, return_attention_weights=True)
        else:
            if edge_attr is not None:
                h_att1 = self.gat1(h, edge_index, edge_attr=edge_attr)
            else:
                h_att1 = self.gat1(h, edge_index)
            att_weights1 = None

        h = self.norm2(h + h_att1)
        h = F.elu(h)
        h = self.dropout(h)

        if edge_attr is not None:
            h_att2 = self.gat2(h, edge_index, edge_attr=edge_attr)
        else:
            h_att2 = self.gat2(h, edge_index)
            
        h = self.norm3(h + h_att2)
        h = F.elu(h)

        logits = self.classifier(h).squeeze(-1)

        if return_attention:
            return logits, att_weights1
        return logits, None
