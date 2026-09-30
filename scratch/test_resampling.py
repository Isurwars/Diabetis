import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE
import lightgbm as lgb
from catboost import CatBoostClassifier

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.config import load_config
from src.data.database import load_raw_cohort
from src.features.preprocessor import FeaturePreprocessor
from src.graph.builder import create_clustered_splits

def main():
    cfg = load_config("configs/default.yaml")
    print("Loading cohort and 43 engineered features...")
    df = load_raw_cohort(cfg.data.db_path, cfg.data.exclude_gestational)
    preprocessor = FeaturePreprocessor(cfg.features)
    X, y, upm_clusters, sample_weights = preprocessor.fit_transform(df)

    n_nodes = len(X)
    train_mask, val_mask, test_mask = create_clustered_splits(upm_clusters, n_nodes, cfg.split)
    train_idx = np.where(train_mask.numpy())[0]
    val_idx = np.where(val_mask.numpy())[0]
    test_idx = np.where(test_mask.numpy())[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Original Training Set: {len(y_train):,} (Positives: {y_train.sum():,}, {y_train.mean():.2%})")
    print(f"Test Set (Natural Prevalence): {len(y_test):,} (Positives: {y_test.sum():,}, {y_test.mean():.2%})")

    # Baseline without resampling (LightGBM & CatBoost)
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    lgb_base = lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=42, verbose=-1)
    lgb_base.fit(X_train, y_train)
    p_lgb_base = lgb_base.predict_proba(X_test)[:, 1]
    print(f"\n[Baseline - No Resampling]: LightGBM Test ROC = {roc_auc_score(y_test, p_lgb_base):.4f}, PR = {average_precision_score(y_test, p_lgb_base):.4f}")

    # 1. Test Borderline-SMOTE
    print("\n[Testing Borderline-SMOTE on Training Split]...")
    bsmote = BorderlineSMOTE(sampling_strategy=0.35, random_state=42)
    X_train_bs, y_train_bs = bsmote.fit_resample(X_train, y_train)
    print(f"  Resampled Train: {len(y_train_bs):,} (Positives: {y_train_bs.sum():,}, {y_train_bs.mean():.2%})")

    lgb_bs = lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_bs.fit(X_train_bs, y_train_bs)
    p_lgb_bs = lgb_bs.predict_proba(X_test)[:, 1]
    print(f"  LightGBM (Borderline-SMOTE) -> Test ROC = {roc_auc_score(y_test, p_lgb_bs):.4f}, PR = {average_precision_score(y_test, p_lgb_bs):.4f}, Brier = {brier_score_loss(y_test, p_lgb_bs):.4f}")

    # 2. Test SMOTE-ENN
    print("\n[Testing SMOTE-ENN on Training Split]...")
    smote_enn = SMOTEENN(sampling_strategy=0.35, random_state=42)
    X_train_sme, y_train_sme = smote_enn.fit_resample(X_train, y_train)
    print(f"  Resampled Train: {len(y_train_sme):,} (Positives: {y_train_sme.sum():,}, {y_train_sme.mean():.2%})")

    lgb_sme = lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_sme.fit(X_train_sme, y_train_sme)
    p_lgb_sme = lgb_sme.predict_proba(X_test)[:, 1]
    print(f"  LightGBM (SMOTE-ENN)        -> Test ROC = {roc_auc_score(y_test, p_lgb_sme):.4f}, PR = {average_precision_score(y_test, p_lgb_sme):.4f}, Brier = {brier_score_loss(y_test, p_lgb_sme):.4f}")

    # CatBoost on Borderline-SMOTE
    cat_bs = CatBoostClassifier(iterations=250, depth=6, learning_rate=0.04, eval_metric="AUC", random_seed=42, verbose=0)
    cat_bs.fit(X_train_bs, y_train_bs)
    p_cat_bs = cat_bs.predict_proba(X_test)[:, 1]
    print(f"  CatBoost (Borderline-SMOTE) -> Test ROC = {roc_auc_score(y_test, p_cat_bs):.4f}, PR = {average_precision_score(y_test, p_cat_bs):.4f}, Brier = {brier_score_loss(y_test, p_cat_bs):.4f}")

    # Blending Resampled + Base
    blend_probs = 0.5 * p_lgb_base + 0.5 * p_lgb_bs
    print(f"\n[Hybrid Blend: Base + Resampled] -> Test ROC = {roc_auc_score(y_test, blend_probs):.4f}, PR = {average_precision_score(y_test, blend_probs):.4f}")

if __name__ == "__main__":
    main()
