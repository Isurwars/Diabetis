import numpy as np
import torch
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from ..config import GraphConfig, SplitConfig

def build_knn_graph(X: np.ndarray, config: GraphConfig = None) -> torch.Tensor:
    """
    Constructs an undirected k-NN patient similarity graph based on feature distance.
    Returns: edge_index tensor of shape (2, num_edges).
    """
    cfg = config or GraphConfig()
    nn_model = NearestNeighbors(
        n_neighbors=cfg.k + 1,
        metric=cfg.metric,
        algorithm=cfg.algorithm,
        n_jobs=-1
    )
    nn_model.fit(X)
    
    # Query k nearest neighbors (excluding self-loop at column 0)
    _, indices = nn_model.kneighbors(X)
    
    num_nodes = len(X)
    sources = np.repeat(np.arange(num_nodes), cfg.k)
    targets = indices[:, 1:].flatten()
    
    # Bidirectional edges
    edge_index = np.vstack([
        np.concatenate([sources, targets]),
        np.concatenate([targets, sources])
    ])
    
    # Deduplicate edges
    edge_index = np.unique(edge_index, axis=1)
    return torch.tensor(edge_index, dtype=torch.long)


def build_community_edges(upm_clusters: np.ndarray) -> np.ndarray:
    """Connects nodes residing in the exact same UPM / community."""
    nodes_by_upm = {}
    for idx, u in enumerate(upm_clusters):
        nodes_by_upm.setdefault(u, []).append(idx)
        
    src_list, dst_list = [], []
    for u, members in nodes_by_upm.items():
        n_m = len(members)
        if n_m > 1:
            for i in range(n_m):
                for j in range(i + 1, n_m):
                    src_list.extend([members[i], members[j]])
                    dst_list.extend([members[j], members[i]])
                    
    if not src_list:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack([src_list, dst_list])


def build_multirelational_graph(
    X: np.ndarray,
    upm_clusters: np.ndarray,
    config: GraphConfig = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs a multi-relational graph:
    - Relation 0: Clinical similarity (k-NN)
    - Relation 1: Community / Local geographic co-habitation (same UPM)
    Returns: (edge_index, edge_type)
    """
    knn_edges = build_knn_graph(X, config).numpy()
    knn_types = np.zeros(knn_edges.shape[1], dtype=np.int64)

    comm_edges = build_community_edges(upm_clusters)
    comm_types = np.ones(comm_edges.shape[1], dtype=np.int64)

    all_edges = np.hstack([knn_edges, comm_edges])
    all_types = np.concatenate([knn_types, comm_types])

    return torch.tensor(all_edges, dtype=torch.long), torch.tensor(all_types, dtype=torch.long)


def create_clustered_splits(
    upm_clusters: np.ndarray,
    n_nodes: int,
    config: SplitConfig = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Partitions nodes into train/val/test by UPM sampling cluster
    to guarantee zero spatial data snooping.
    """
    cfg = config or SplitConfig()
    
    # Split Train vs (Val + Test)
    gss1 = GroupShuffleSplit(
        n_splits=1,
        train_size=cfg.train_ratio,
        random_state=cfg.random_state
    )
    train_idx, temp_idx = next(gss1.split(np.arange(n_nodes), groups=upm_clusters))
    
    # Split Temp into Val and Test
    temp_groups = upm_clusters[temp_idx]
    relative_val_size = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_val_size,
        random_state=cfg.random_state
    )
    val_sub_idx, test_sub_idx = next(gss2.split(temp_idx, groups=temp_groups))
    
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask
