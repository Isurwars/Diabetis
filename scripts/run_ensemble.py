#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
from torch_geometric.data import Data

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.config import load_config
from src.data.database import load_raw_cohort
from src.features.preprocessor import FeaturePreprocessor
from src.graph.builder import build_knn_graph, create_clustered_splits
from src.models.gatv2 import GATv2DiabetesClassifier
from src.models.baselines import TabularBaselines
from src.models.ensemble import TabularGraphEnsemble
from src.evaluation.metrics import evaluate_predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate LightGBM + GATv2 Ensembles on Held-out UPM Clusters")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_gatv2.pt", help="Path to GATv2 weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 72)
    print("      ENSANUT 2018 TABULAR + GRAPH GNN ENSEMBLE BENCHMARK        ")
    print("=" * 72)

    # 1. Load Data & Preprocessing
    print("\n[1/5] Extracting cohort and preprocessing features...")
    df = load_raw_cohort(cfg.data.db_path, cfg.data.exclude_gestational)
    preprocessor = FeaturePreprocessor(cfg.features)
    X, y, upm_clusters, sample_weights = preprocessor.fit_transform(df)
    n_nodes = len(X)
    in_features = X.shape[1]

    # 2. Clustered Split (UPM holdout)
    train_mask, val_mask, test_mask = create_clustered_splits(upm_clusters, n_nodes, cfg.split)
    train_idx = np.where(train_mask.numpy())[0]
    val_idx = np.where(val_mask.numpy())[0]
    test_idx = np.where(test_mask.numpy())[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    train_weights = sample_weights[train_idx]

    print(f"      Cohorts: Train={len(train_idx):,}, Val={len(val_idx):,}, Test={len(test_idx):,} (Clustered by UPM)")

    # 3. Train Tabular Baselines (LightGBM & XGBoost)
    print("\n[2/5] Fitting Tabular Models (LightGBM & XGBoost)...")
    lgb_model, _, _ = TabularBaselines.train_lightgbm(X_train, y_train, X_test, y_test)
    lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test_probs = lgb_model.predict_proba(X_test)[:, 1]

    xgb_model, _, _ = TabularBaselines.train_xgboost(X_train, y_train, X_test, y_test, sample_weights=train_weights)
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]

    # 4. Load / Run GATv2
    print("\n[3/5] Loading GATv2 model and computing graph representations...")
    edge_index = build_knn_graph(X, cfg.graph)
    
    device = torch.device("cpu")
    gat_model = GATv2DiabetesClassifier(in_features, cfg.model, edge_dim=None).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"      Loading weights from: {args.checkpoint}")
        gat_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"      [Warning] Checkpoint not found at {args.checkpoint}. Please train first.")
        return

    gat_model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32)
        logits, _ = gat_model(x_tensor, edge_index)
        all_gat_probs = torch.sigmoid(logits).cpu().numpy()
        h_all = gat_model.get_embeddings(x_tensor, edge_index).cpu().numpy()

    gat_val_probs = all_gat_probs[val_idx]
    gat_test_probs = all_gat_probs[test_idx]

    h_train = h_all[train_idx]
    h_val = h_all[val_idx]
    h_test = h_all[test_idx]

    # 5. Evaluate Individual Models
    m_lgb = evaluate_predictions(y_test, lgb_test_probs)
    m_xgb = evaluate_predictions(y_test, xgb_test_probs)
    m_gat = evaluate_predictions(y_test, gat_test_probs)

    print("\n[4/5] Evaluating Individual Base Models on Test UPM Clusters:")
    print(f"  • LightGBM alone       : ROC-AUC = {m_lgb['roc_auc']:.4f} | PR-AUC = {m_lgb['pr_auc']:.4f} | Brier = {m_lgb['brier_score']:.4f}")
    print(f"  • XGBoost alone        : ROC-AUC = {m_xgb['roc_auc']:.4f} | PR-AUC = {m_xgb['pr_auc']:.4f} | Brier = {m_xgb['brier_score']:.4f}")
    print(f"  • GATv2 alone (k-NN)   : ROC-AUC = {m_gat['roc_auc']:.4f} | PR-AUC = {m_gat['pr_auc']:.4f} | Brier = {m_gat['brier_score']:.4f}")

    # 6. Evaluate Ensemble Strategies
    print("\n[5/5] Building and Benchmarking Ensembles...")

    # Strategy A: Optimal Validation Blend (LightGBM + GATv2)
    best_beta, val_roc = TabularGraphEnsemble.find_optimal_blend(lgb_val_probs, gat_val_probs, y_val, metric="roc_auc")
    blend_test_probs = best_beta * lgb_test_probs + (1.0 - best_beta) * gat_test_probs
    m_blend = evaluate_predictions(y_test, blend_test_probs)
    print(f"\n  [Ensemble 1] Probability Blend (LightGBM * {best_beta:.2f} + GATv2 * {1.0-best_beta:.2f}):")
    print(f"      • Test ROC-AUC : {m_blend['roc_auc']:.4f}  (vs LightGBM: {m_blend['roc_auc'] - m_lgb['roc_auc']:+.4f})")
    print(f"      • Test PR-AUC  : {m_blend['pr_auc']:.4f}  (vs LightGBM: {m_blend['pr_auc'] - m_lgb['pr_auc']:+.4f})")
    print(f"      • Brier Score  : {m_blend['brier_score']:.4f}")
    print(f"      • Sensitivity (Youden) : {m_blend['report_youden']['Diagnosed Diabetes']['recall']*100:.2f}%")

    # Strategy B: Tri-Model Average Blend (LGBM + XGB + GATv2)
    tri_test_probs = (lgb_test_probs + xgb_test_probs + gat_test_probs) / 3.0
    m_tri = evaluate_predictions(y_test, tri_test_probs)
    print(f"\n  [Ensemble 2] Tri-Model Equal Blend (LGBM + XGB + GATv2):")
    print(f"      • Test ROC-AUC : {m_tri['roc_auc']:.4f}")
    print(f"      • Test PR-AUC  : {m_tri['pr_auc']:.4f}")
    print(f"      • Brier Score  : {m_tri['brier_score']:.4f}")

    # Strategy C: Stacking Meta-Learner (Logistic Regression on [p_lgb, p_xgb, p_gat])
    V_val = np.column_stack([lgb_val_probs, xgb_val_probs, gat_val_probs])
    V_test = np.column_stack([lgb_test_probs, xgb_test_probs, gat_test_probs])
    meta_model, meta_test_probs, _, _ = TabularGraphEnsemble.train_meta_learner(V_val, y_val, V_test, y_test)
    m_meta = evaluate_predictions(y_test, meta_test_probs)
    print(f"\n  [Ensemble 3] Stacking Meta-Learner (Logistic Regression on Val Predictions):")
    print(f"      • Test ROC-AUC : {m_meta['roc_auc']:.4f}")
    print(f"      • Test PR-AUC  : {m_meta['pr_auc']:.4f}")
    print(f"      • Brier Score  : {m_meta['brier_score']:.4f}")

    # Strategy D: GNN Latent Embedding Fusion [X || h_GNN] -> LightGBM
    print(f"\n  [Ensemble 4] Latent Embedding Fusion: LightGBM trained on [Tabular X || GNN h]...")
    _, stacked_test_probs, _, _ = TabularGraphEnsemble.train_gnn_stacked_lightgbm(
        X_train, h_train, y_train, X_test, h_test, y_test
    )
    m_stacked = evaluate_predictions(y_test, stacked_test_probs)
    print(f"      • Test ROC-AUC : {m_stacked['roc_auc']:.4f}  (vs LightGBM: {m_stacked['roc_auc'] - m_lgb['roc_auc']:+.4f})")
    print(f"      • Test PR-AUC  : {m_stacked['pr_auc']:.4f}  (vs LightGBM: {m_stacked['pr_auc'] - m_lgb['pr_auc']:+.4f})")
    print(f"      • Brier Score  : {m_stacked['brier_score']:.4f}")

    print("\n" + "=" * 72)
    print("                    FINAL BENCHMARK COMPARISON TABLE                    ")
    print("=" * 72)
    print(f"{'Model / Architecture':<42} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Brier':<8}")
    print("-" * 72)
    print(f"{'GATv2 (Clinical k-NN)':<42} | {m_gat['roc_auc']:<8.4f} | {m_gat['pr_auc']:<8.4f} | {m_gat['brier_score']:<8.4f}")
    print(f"{'XGBoost (Weighted)':<42} | {m_xgb['roc_auc']:<8.4f} | {m_xgb['pr_auc']:<8.4f} | {m_xgb['brier_score']:<8.4f}")
    print(f"{'LightGBM Baseline':<42} | {m_lgb['roc_auc']:<8.4f} | {m_lgb['pr_auc']:<8.4f} | {m_lgb['brier_score']:<8.4f}")
    print("-" * 72)
    print(f"{'Ensemble 1: LGBM + GATv2 Prob Blend':<42} | {m_blend['roc_auc']:<8.4f} | {m_blend['pr_auc']:<8.4f} | {m_blend['brier_score']:<8.4f}")
    print(f"{'Ensemble 2: Tri-Model Blend (LGB+XGB+GAT)':<42} | {m_tri['roc_auc']:<8.4f} | {m_tri['pr_auc']:<8.4f} | {m_tri['brier_score']:<8.4f}")
    print(f"{'Ensemble 3: Stacking Meta-Learner (LR)':<42} | {m_meta['roc_auc']:<8.4f} | {m_meta['pr_auc']:<8.4f} | {m_meta['brier_score']:<8.4f}")
    print(f"{'Ensemble 4: [Tabular X || GNN h] -> LightGBM':<42} | {m_stacked['roc_auc']:<8.4f} | {m_stacked['pr_auc']:<8.4f} | {m_stacked['brier_score']:<8.4f}")
    print("=" * 72)

if __name__ == "__main__":
    main()
