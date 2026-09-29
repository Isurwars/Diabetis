#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.config import load_config
from src.data.database import load_raw_cohort
from src.features.preprocessor import FeaturePreprocessor
from src.graph.builder import create_clustered_splits
from src.models.baselines import TabularBaselines

def main():
    parser = argparse.ArgumentParser(description="Benchmark tabular ML baselines against GNN")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("=" * 65)
    print("        TABULAR BASELINES BENCHMARK (XGBoost, LightGBM, LR)     ")
    print("=" * 65)

    # 1. Load data
    df = load_raw_cohort(cfg.data.db_path, cfg.data.exclude_gestational)
    preprocessor = FeaturePreprocessor(cfg.features)
    X, y, upm_clusters, sample_weights = preprocessor.fit_transform(df)
    n_nodes = len(X)

    # 2. Clustered Split (Same as GNN)
    train_mask, val_mask, test_mask = create_clustered_splits(upm_clusters, n_nodes, cfg.split)
    
    train_idx = np.where(train_mask.numpy())[0]
    test_idx = np.where(test_mask.numpy())[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    train_weights = sample_weights[train_idx]

    print(f"Dataset: Train={len(X_train):,}, Test={len(X_test):,} (Clustered by UPM)\n")

    # 3. Logistic Regression
    print("[1/3] Training Logistic Regression Baseline...")
    _, lr_roc, lr_pr = TabularBaselines.train_logistic_regression(X_train, y_train, X_test, y_test)
    print(f"      Logistic Regression -> Test ROC-AUC: {lr_roc:.4f} | Test PR-AUC: {lr_pr:.4f}")

    # 4. LightGBM
    print("[2/3] Training LightGBM Baseline...")
    _, lgb_roc, lgb_pr = TabularBaselines.train_lightgbm(X_train, y_train, X_test, y_test)
    print(f"      LightGBM            -> Test ROC-AUC: {lgb_roc:.4f} | Test PR-AUC: {lgb_pr:.4f}")

    # 5. XGBoost
    print("[3/3] Training XGBoost Baseline...")
    _, xgb_roc, xgb_pr = TabularBaselines.train_xgboost(X_train, y_train, X_test, y_test, sample_weights=train_weights)
    print(f"      XGBoost             -> Test ROC-AUC: {xgb_roc:.4f} | Test PR-AUC: {xgb_pr:.4f}")

    print("\n" + "=" * 65)
    print("                   BASELINE COMPARISON TABLE                 ")
    print("=" * 65)
    print(f"  • Logistic Regression  : ROC-AUC = {lr_roc:.4f} | PR-AUC = {lr_pr:.4f}")
    print(f"  • LightGBM             : ROC-AUC = {lgb_roc:.4f} | PR-AUC = {lgb_pr:.4f}")
    print(f"  • XGBoost (Weighted)   : ROC-AUC = {xgb_roc:.4f} | PR-AUC = {xgb_pr:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
