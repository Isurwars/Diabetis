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
    parser = argparse.ArgumentParser(description="Evaluate Quad-Model Super-Learner (CatBoost, LightGBM, XGBoost, GATv2) on Held-out UPM Clusters")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_gatv2.pt", help="Path to GATv2 weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 75)
    print("      ENSANUT 2018 QUAD-MODEL SUPER-LEARNER BENCHMARK (CatBoost+GNN)      ")
    print("=" * 75)

    # 1. Load Data & Preprocessing
    print("\n[1/5] Extracting cohort and preprocessing 43 engineered features...")
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

    # 3. Train Tabular Models (LightGBM, XGBoost, CatBoost)
    print("\n[2/5] Fitting Tabular Models (LightGBM, XGBoost, CatBoost)...")
    
    # LightGBM
    lgb_model, _, _ = TabularBaselines.train_lightgbm(X_train, y_train, X_test, y_test)
    lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test_probs = lgb_model.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb_model, _, _ = TabularBaselines.train_xgboost(X_train, y_train, X_test, y_test, sample_weights=train_weights)
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]

    # CatBoost (Standard Weighted)
    cat_model, _, _ = TabularBaselines.train_catboost(X_train, y_train, X_test, y_test, sample_weights=train_weights)
    cat_val_probs = cat_model.predict_proba(X_val)[:, 1]
    cat_test_probs = cat_model.predict_proba(X_test)[:, 1]

    # CatBoost (Borderline-SMOTE Boundary Cleaned)
    from imblearn.over_sampling import BorderlineSMOTE
    bsmote = BorderlineSMOTE(sampling_strategy=0.35, random_state=42)
    X_train_bs, y_train_bs = bsmote.fit_resample(X_train, y_train)
    cat_bs_model, _, _ = TabularBaselines.train_catboost(X_train_bs, y_train_bs, X_test, y_test)
    cat_bs_val_probs = cat_bs_model.predict_proba(X_val)[:, 1]
    cat_bs_test_probs = cat_bs_model.predict_proba(X_test)[:, 1]

    # 4. Load / Run Upgraded GATv2 (Ego-Skip + DropEdge)
    print("\n[3/5] Loading Upgraded GATv2 model (Ego-Skip + DropEdge)...")
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

    gat_val_probs = all_gat_probs[val_idx]
    gat_test_probs = all_gat_probs[test_idx]

    # 5. Evaluate Individual Models
    m_lgb = evaluate_predictions(y_test, lgb_test_probs)
    m_xgb = evaluate_predictions(y_test, xgb_test_probs)
    m_cat = evaluate_predictions(y_test, cat_test_probs)
    m_cat_bs = evaluate_predictions(y_test, cat_bs_test_probs)
    m_gat = evaluate_predictions(y_test, gat_test_probs)

    print("\n[4/5] Evaluating Individual Base Models on Test UPM Clusters:")
    print(f"  • LightGBM alone              : ROC-AUC = {m_lgb['roc_auc']:.4f} | PR-AUC = {m_lgb['pr_auc']:.4f} | Brier = {m_lgb['brier_score']:.4f}")
    print(f"  • XGBoost alone               : ROC-AUC = {m_xgb['roc_auc']:.4f} | PR-AUC = {m_xgb['pr_auc']:.4f} | Brier = {m_xgb['brier_score']:.4f}")
    print(f"  • CatBoost alone              : ROC-AUC = {m_cat['roc_auc']:.4f} | PR-AUC = {m_cat['pr_auc']:.4f} | Brier = {m_cat['brier_score']:.4f}")
    print(f"  • CatBoost (Borderline-SMOTE) : ROC-AUC = {m_cat_bs['roc_auc']:.4f} | PR-AUC = {m_cat_bs['pr_auc']:.4f} | Brier = {m_cat_bs['brier_score']:.4f}")
    print(f"  • GATv2 (Ego-Skip + DropEdge) : ROC-AUC = {m_gat['roc_auc']:.4f} | PR-AUC = {m_gat['pr_auc']:.4f} | Brier = {m_gat['brier_score']:.4f}")

    # 6. Evaluate Ensemble Strategies
    print("\n[5/5] Building and Benchmarking Ensembles...")

    # Strategy 1: Quad-Model Equal Blend (LGB + XGB + CatBoost_BS + GATv2)
    quad_equal_test_probs = (lgb_test_probs + xgb_test_probs + cat_bs_test_probs + gat_test_probs) / 4.0
    m_quad_eq = evaluate_predictions(y_test, quad_equal_test_probs)
    print(f"\n  [Ensemble 1] Quad-Model Equal Blend (LGB + XGB + CatBoost_BS + GATv2):")
    print(f"      • Test ROC-AUC : {m_quad_eq['roc_auc']:.4f} | Test PR-AUC : {m_quad_eq['pr_auc']:.4f} | Brier : {m_quad_eq['brier_score']:.4f}")

    # Strategy 2: Quad-Model Optimal Simplex Blend
    val_probs_mat = np.column_stack([lgb_val_probs, xgb_val_probs, cat_bs_val_probs, gat_val_probs])
    test_probs_mat = np.column_stack([lgb_test_probs, xgb_test_probs, cat_bs_test_probs, gat_test_probs])
    opt_weights = TabularGraphEnsemble.find_optimal_multimodel_blend(val_probs_mat, y_val, metric="roc_auc")
    quad_opt_test_probs = np.dot(test_probs_mat, opt_weights)
    m_quad_opt = evaluate_predictions(y_test, quad_opt_test_probs)
    print(f"\n  [Ensemble 2] Quad-Model Simplex Blend (Weights: LGB={opt_weights[0]:.2f}, XGB={opt_weights[1]:.2f}, Cat_BS={opt_weights[2]:.2f}, GAT={opt_weights[3]:.2f}):")
    print(f"      • Test ROC-AUC : {m_quad_opt['roc_auc']:.4f} | Test PR-AUC : {m_quad_opt['pr_auc']:.4f} | Brier : {m_quad_opt['brier_score']:.4f}")

    # Strategy 3: Quad-Model Stacking Super-Learner (L2 Regularized Logistic Regression)
    meta_model, meta_test_probs, _, _ = TabularGraphEnsemble.train_meta_learner(val_probs_mat, y_val, test_probs_mat, y_test, C=1.0)
    m_meta = evaluate_predictions(y_test, meta_test_probs)
    print(f"\n  [Ensemble 3] Quad-Model Stacking Super-Learner (L2 Regularized Logistic Regression):")
    print(f"      • Test ROC-AUC : {m_meta['roc_auc']:.4f} | Test PR-AUC : {m_meta['pr_auc']:.4f} | Brier : {m_meta['brier_score']:.4f}")
    print(f"      • Screening Recall (Youden): {m_meta['report_youden']['Diagnosed Diabetes']['recall']*100:.2f}%")


    print("\n" + "=" * 75)
    print("                    QUAD-MODEL FINAL BENCHMARK SUMMARY                    ")
    print("=" * 75)
    print(f"{'Model / Architecture':<45} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Brier':<8}")
    print("-" * 75)
    print(f"{'GATv2 (Clinical k-NN)':<45} | {m_gat['roc_auc']:<8.4f} | {m_gat['pr_auc']:<8.4f} | {m_gat['brier_score']:<8.4f}")
    print(f"{'XGBoost (Weighted)':<45} | {m_xgb['roc_auc']:<8.4f} | {m_xgb['pr_auc']:<8.4f} | {m_xgb['brier_score']:<8.4f}")
    print(f"{'CatBoost (Weighted)':<45} | {m_cat['roc_auc']:<8.4f} | {m_cat['pr_auc']:<8.4f} | {m_cat['brier_score']:<8.4f}")
    print(f"{'CatBoost (Borderline-SMOTE)':<45} | {m_cat_bs['roc_auc']:<8.4f} | {m_cat_bs['pr_auc']:<8.4f} | {m_cat_bs['brier_score']:<8.4f}")
    print(f"{'LightGBM Baseline':<45} | {m_lgb['roc_auc']:<8.4f} | {m_lgb['pr_auc']:<8.4f} | {m_lgb['brier_score']:<8.4f}")
    print("-" * 75)
    print(f"{'Ensemble 1: Quad-Model Equal Blend':<45} | {m_quad_eq['roc_auc']:<8.4f} | {m_quad_eq['pr_auc']:<8.4f} | {m_quad_eq['brier_score']:<8.4f}")
    print(f"{'Ensemble 2: Quad-Model Simplex Blend':<45} | {m_quad_opt['roc_auc']:<8.4f} | {m_quad_opt['pr_auc']:<8.4f} | {m_quad_opt['brier_score']:<8.4f}")
    print(f"{'Ensemble 3: Quad Super-Learner (L2-LR)':<45} | {m_meta['roc_auc']:<8.4f} | {m_meta['pr_auc']:<8.4f} | {m_meta['brier_score']:<8.4f}")

    print("=" * 75)

if __name__ == "__main__":
    main()
