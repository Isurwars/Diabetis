import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class PatientAttentionExplainer:
    """Inspects GATv2 attention weights to explain individual patient risk."""
    
    def __init__(self, model, data, feature_names: List[str]):
        self.model = model
        self.data = data
        self.feature_names = feature_names

    def explain_patient(self, node_idx: int, top_n: int = 5) -> Dict[str, Any]:
        """Returns the top-n neighbor peers with their attention weights, relation types, and risk traits."""
        self.model.eval()
        edge_type = getattr(self.data, "edge_type", None)

        with torch.no_grad():
            logits, (edge_index, alpha) = self.model(
                self.data.x, self.data.edge_index, edge_type=edge_type, return_attention=True
            )
            prob = torch.sigmoid(logits[node_idx]).item()

        # Find incoming edges where target is node_idx
        target_mask = (edge_index[1] == node_idx)
        neighbor_nodes = edge_index[0][target_mask].cpu().numpy()
        # Average attention weights across heads
        neighbor_weights = alpha[target_mask].mean(dim=-1).cpu().numpy()
        
        # Handle self-loops added by GATv2Conv
        if edge_type is not None:
            num_added = edge_index.shape[1] - len(edge_type)
            if num_added > 0:
                pad_types = torch.full((num_added,), fill_value=-1, dtype=edge_type.dtype, device=edge_type.device)
                full_edge_type = torch.cat([edge_type, pad_types])
            else:
                full_edge_type = edge_type
            rel_types = full_edge_type[target_mask].cpu().numpy()
        else:
            rel_types = np.zeros(len(neighbor_nodes), dtype=int)

        sorted_indices = np.argsort(neighbor_weights)[::-1]
        top_peers = []

        for idx in sorted_indices:
            peer_node = int(neighbor_nodes[idx])
            # Skip self-loops in peer explanations
            if peer_node == node_idx:
                continue

            weight = float(neighbor_weights[idx])
            peer_label = int(self.data.y[peer_node].item())
            peer_features = self.data.x[peer_node].cpu().numpy()
            relation_str = "Local Community (Same UPM)" if rel_types[idx] == 1 else "Clinical k-NN Peer"

            top_peers.append({
                "peer_node_id": peer_node,
                "attention_weight": weight,
                "relation_type": relation_str,
                "peer_is_diabetic": bool(peer_label == 1),
                "features": {name: float(val) for name, val in zip(self.feature_names, peer_features)}
            })

            if len(top_peers) >= top_n:
                break

        return {
            "patient_node_id": node_idx,
            "predicted_risk": prob,
            "actual_label": int(self.data.y[node_idx].item()),
            "top_influential_peers": top_peers
        }
