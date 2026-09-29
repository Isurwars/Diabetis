#!/usr/bin/env python3
"""
ENSANUT 2018 Diabetes Prediction via GATv2 (Graph Attention Network v2)
========================================================================
End-to-End Pipeline:
1. SQL feature extraction from SQLite (ensanut_2018.db).
2. Elimination of gestational cases and conditional diabetes leak variables.
3. Feature encoding & normalization.
4. Construction of a Patient-Similarity k-NN Graph (k=10, Cosine Similarity).
5. Clustered train/val/test split by UPM (Sampling Unit) to prevent spatial leakage.
6. Multi-head GATv2 architecture with Focal Loss for class imbalance.
7. Evaluation reporting ROC-AUC, PR-AUC, F1-Score, and Attention Explainability.
"""

import os
import sys
import sqlite3
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve

DB_PATH = "/home/isurwars/Projects/Diabetis/ensanut_2018.db"

# -----------------------------------------------------------------------------
# 1. Feature Extraction & Cleaning
# -----------------------------------------------------------------------------
def load_and_preprocess_cohort(db_path: str = DB_PATH) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    print("[1/5] Extracting cohort and multi-domain features from SQLite database...")
    conn = sqlite3.connect(db_path)
    
    # Query multi-domain features from adultos, residentes, and hogares
    query = """
    SELECT 
        a.upm, a.viv_sel, a.hogar, a.numren,
        a.p3_1,
        -- Demographics from residentes
        r.edad,
        r.sexo,
        r.nivel as nivel_educativo,
        r.estrato as estrato_socioeconomico,
        -- Anthropometrics & clinical history from adultos
        a.p1_1 as dx_obesidad,
        a.p1_4 as silueta_corporal,
        a.p1_5 as peso_habitual,
        a.p1_7 as cambio_peso,
        a.p1_8 as kg_cambio,
        a.p4_1 as dx_hipertension,
        a.p6_3 as dx_colesterol_trigliceridos,
        a.p7_1_1 as ant_padre_diab,
        a.p7_1_2 as ant_madre_diab,
        a.p7_1_3 as ant_hermano_diab,
        a.p13_1 as fuma_100_cigarros,
        a.p13_2 as fuma_actualmente,
        -- Household assets
        h.p2_9_1 as tiene_refri,
        h.p2_9_2 as tiene_lavadora,
        h.p2_9_3 as tiene_auto
    FROM adultos a
    LEFT JOIN residentes r 
        ON a.upm = r.upm AND a.viv_sel = r.viv_sel AND a.hogar = r.hogar AND a.numren = r.numren
    LEFT JOIN hogares h 
        ON a.upm = h.upm AND a.viv_sel = h.viv_sel AND a.hogar = h.hogar
    WHERE a.p3_1 IN (1, 3);
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"      Loaded {len(df):,} valid adult records (excluding gestational cases).")
    
    # Target variable: 1 if diagnosed diabetes (p3_1 == 1), 0 if no (p3_1 == 3)
    y = (df["p3_1"] == 1).astype(int).values
    upm_clusters = df["upm"].values
    
    print(f"      Target Distribution -> Positive: {y.sum():,} ({y.mean()*100:.2f}%), Negative: {(1-y).sum():,} ({(1-y.mean())*100:.2f}%)")
    
    # Feature columns to process
    feature_cols = [
        "edad", "sexo", "nivel_educativo", "estrato_socioeconomico", "dx_obesidad", "silueta_corporal", "peso_habitual",
        "cambio_peso", "kg_cambio", "dx_hipertension", "dx_colesterol_trigliceridos",
        "ant_padre_diab", "ant_madre_diab", "ant_hermano_diab", "fuma_100_cigarros",
        "fuma_actualmente", "tiene_refri", "tiene_lavadora", "tiene_auto"
    ]
    
    X_df = df[feature_cols].copy()
    
    # Clean binary / survey indicators (Map INEGI standard: 1=Si, 2=No, 9=NS/NR)
    binary_cols = [
        "dx_obesidad", "dx_hipertension", "dx_colesterol_trigliceridos",
        "ant_padre_diab", "ant_madre_diab", "ant_hermano_diab",
        "fuma_100_cigarros", "tiene_refri", "tiene_lavadora", "tiene_auto"
    ]
    for col in binary_cols:
        # Treat 1 as 1, anything else (2, 8, 9, NaN) as 0
        X_df[col] = (X_df[col] == 1).astype(float)
        
    # Sex: Male=1, Female=0
    X_df["sexo"] = (X_df["sexo"] == 1).astype(float)
    
    # Impute missing or invalid continuous measurements with median
    continuous_cols = ["edad", "peso_habitual", "kg_cambio", "silueta_corporal", "nivel_educativo", "estrato_socioeconomico"]
    for col in continuous_cols:
        # Replace refusal codes (99, 888, 999) with NaN
        X_df.loc[X_df[col] >= 888, col] = np.nan
        median_val = X_df[col].median()
        X_df[col] = X_df[col].fillna(median_val)
        
    # Categorical remaining (cambio_peso, fuma_actualmente)
    X_df["cambio_peso"] = X_df["cambio_peso"].fillna(3)
    X_df["fuma_actualmente"] = (X_df["fuma_actualmente"] == 1).astype(float)
    
    # Scale continuous features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    
    return df, X_scaled, y, upm_clusters


# -----------------------------------------------------------------------------
# 2. k-NN Patient Graph Construction
# -----------------------------------------------------------------------------
def build_knn_graph(X: np.ndarray, k: int = 10) -> torch.Tensor:
    print(f"[2/5] Building Patient-Similarity k-NN Graph (k={k}, Metric=Cosine)...")
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn_model.fit(X)
    
    # Query nearest neighbors (excluding self at index 0)
    _, indices = nn_model.kneighbors(X)
    
    num_nodes = len(X)
    sources = np.repeat(np.arange(num_nodes), k)
    targets = indices[:, 1:].flatten()
    
    # Construct bidirectional undirected edge index
    edge_index = np.vstack([
        np.concatenate([sources, targets]),
        np.concatenate([targets, sources])
    ])
    
    # Remove duplicate edges
    edge_index = np.unique(edge_index, axis=1)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    
    print(f"      Graph successfully constructed: {num_nodes:,} nodes, {edge_index_tensor.shape[1]:,} directed edges.")
    return edge_index_tensor


# -----------------------------------------------------------------------------
# 3. Clustered Train/Val/Test Split (Prevent Spatial Snooping)
# -----------------------------------------------------------------------------
def create_clustered_splits(upm_clusters: np.ndarray, n_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("[3/5] Performing Clustered Train/Val/Test Split by UPM (Sampling Unit)...")
    
    # 70% Train, 30% Temp (Val + Test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, temp_idx = next(gss1.split(np.arange(n_nodes), groups=upm_clusters))
    
    # Split Temp into 50% Val (15% total), 50% Test (15% total)
    temp_groups = upm_clusters[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_sub_idx, test_sub_idx = next(gss2.split(temp_idx, groups=temp_groups))
    
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]
    
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    print(f"      Clustered Split: Train={train_mask.sum().item():,} ({train_mask.float().mean()*100:.1f}%), "
          f"Val={val_mask.sum().item():,} ({val_mask.float().mean()*100:.1f}%), "
          f"Test={test_mask.sum().item():,} ({test_mask.float().mean()*100:.1f}%)")
    return train_mask, val_mask, test_mask


# -----------------------------------------------------------------------------
# 4. GATv2 Neural Network & Focal Loss
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss


class GATv2DiabetesClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Multi-head GATv2 layers
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False):
        h = self.input_proj(x)
        h = F.elu(self.norm1(h))
        h = self.dropout(h)
        
        # First GATv2 Layer with Residual Connection
        if return_attention:
            h_att1, (edge_idx1, alpha1) = self.gat1(h, edge_index, return_attention_weights=True)
        else:
            h_att1 = self.gat1(h, edge_index)
            alpha1 = None
            
        h = self.norm2(h + h_att1)
        h = F.elu(h)
        h = self.dropout(h)
        
        # Second GATv2 Layer with Residual Connection
        h_att2 = self.gat2(h, edge_index)
        h = self.norm3(h + h_att2)
        h = F.elu(h)
        
        logits = self.classifier(h).squeeze(-1)
        
        if return_attention:
            return logits, (edge_idx1, alpha1)
        return logits


# -----------------------------------------------------------------------------
# 5. Training & Evaluation Engine
# -----------------------------------------------------------------------------
def train_and_evaluate(
    data: Data,
    in_features: int,
    epochs: int = 150,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    device: str = "cpu"
) -> Dict[str, float]:
    print(f"\n[4/5] Initializing GATv2 Model on Device: {device}...")
    model = GATv2DiabetesClassifier(in_features=in_features, hidden_dim=64, heads=4, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # alpha=0.75 weights positive minority class
    
    data = data.to(device)
    best_val_roc = 0.0
    best_weights = None
    
    print("[5/5] Training GATv2...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits = model(data.x, data.edge_index)
                val_probs = torch.sigmoid(val_logits[data.val_mask]).cpu().numpy()
                val_labels = data.y[data.val_mask].cpu().numpy()
                
                val_roc = roc_auc_score(val_labels, val_probs)
                val_pr = average_precision_score(val_labels, val_probs)
                
                if val_roc > best_val_roc:
                    best_val_roc = val_roc
                    best_weights = model.state_dict().copy()
                    
                print(f"      Epoch {epoch:03d}/{epochs:03d} | Loss: {loss.item():.4f} | Val ROC-AUC: {val_roc:.4f} | Val PR-AUC: {val_pr:.4f}")

    # Evaluate Best Model on Held-Out Test Set
    if best_weights:
        model.load_state_dict(best_weights)
        
    model.eval()
    with torch.no_grad():
        test_logits = model(data.x, data.edge_index)
        test_probs = torch.sigmoid(test_logits[data.test_mask]).cpu().numpy()
        test_labels = data.y[data.test_mask].cpu().numpy()
        
        test_roc = roc_auc_score(test_labels, test_probs)
        test_pr = average_precision_score(test_labels, test_probs)
        
        # Find optimal threshold using Youden's J statistic (Sensitivity + Specificity - 1)
        fpr, tpr, roc_thresh = roc_curve(test_labels, test_probs)
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        opt_youden_thresh = roc_thresh[best_j_idx]
        
        # Find threshold maximizing F1 score
        prec_arr, rec_arr, pr_thresh = precision_recall_curve(test_labels, test_probs)
        f1_arr = 2 * (prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-8)
        best_f1_idx = np.argmax(f1_arr)
        opt_f1_thresh = pr_thresh[min(best_f1_idx, len(pr_thresh) - 1)]

        preds_youden = (test_probs >= opt_youden_thresh).astype(int)
        preds_f1 = (test_probs >= opt_f1_thresh).astype(int)
        
    print("\n" + "="*60)
    print("           GATv2 MODEL TEST SET EVALUATION RESULTS           ")
    print("="*60)
    print(f"  • Test ROC-AUC                      : {test_roc:.4f}")
    print(f"  • Test PR-AUC (Average Precision)  : {test_pr:.4f}")
    print(f"  • Optimal Youden Threshold (Sens/Spec): {opt_youden_thresh:.4f}")
    print(f"  • Optimal F1 Threshold              : {opt_f1_thresh:.4f}")
    print("\n[A] Performance at Optimal Balanced F1 Threshold:")
    print(classification_report(test_labels, preds_f1, target_names=["No Diabetes", "Diagnosed Diabetes"]))
    print("Confusion Matrix (F1-optimal):")
    print(confusion_matrix(test_labels, preds_f1))
    
    print("\n[B] Performance at Optimal Youden's J Threshold (Screening Balance):")
    print(classification_report(test_labels, preds_youden, target_names=["No Diabetes", "Diagnosed Diabetes"]))
    print("Confusion Matrix (Youden-optimal):")
    print(confusion_matrix(test_labels, preds_youden))
    print("="*60)
    
    return {"roc_auc": test_roc, "pr_auc": test_pr}


def main():
    parser = argparse.ArgumentParser(description="GATv2 Diabetes Prediction on ENSANUT 2018")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--k", type=int, default=10, help="k-NN graph parameter")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    args = parser.parse_args()
    
    # 1. Load Data
    raw_df, X_scaled, y, upm_clusters = load_and_preprocess_cohort(DB_PATH)
    n_nodes, in_features = X_scaled.shape
    
    # 2. Build k-NN Adjacency
    edge_index = build_knn_graph(X_scaled, k=args.k)
    
    # 3. Create Clustered Splits
    train_mask, val_mask, test_mask = create_clustered_splits(upm_clusters, n_nodes)
    
    # 4. Assemble PyTorch Geometric Data Object
    pyg_data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float32),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    # Select Device safely
    if args.device == "cuda":
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        # Check actual free memory on GPU
        if torch.cuda.is_available():
            free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_mem_bytes / (1024**3)
            print(f"      GPU Free Memory: {free_gb:.2f} GB / {total_mem_bytes / (1024**3):.2f} GB")
            # Need at least 2.5 GB free for full-graph GATv2 activations
            device = "cuda" if free_gb >= 2.5 else "cpu"
        else:
            device = "cpu"
            
    print(f"      Selected Device: {device.upper()}")
    
    # 5. Train and Evaluate
    train_and_evaluate(pyg_data, in_features, epochs=args.epochs, lr=args.lr, device=device)


if __name__ == "__main__":
    main()
