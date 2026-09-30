import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

class TabularBaselines:
    """Standard tabular baseline models for benchmarking against GNN."""
    
    @staticmethod
    def train_xgboost(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_weights: np.ndarray = None
    ) -> Tuple[xgb.XGBClassifier, float, float]:
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc"
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        test_probs = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, test_probs)
        pr = average_precision_score(y_test, test_probs)
        return model, roc, pr

    @staticmethod
    def train_logistic_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[LogisticRegression, float, float]:
        model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        test_probs = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, test_probs)
        pr = average_precision_score(y_test, test_probs)
        return model, roc, pr

    @staticmethod
    def train_lightgbm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[lgb.LGBMClassifier, float, float]:
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        test_probs = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, test_probs)
        pr = average_precision_score(y_test, test_probs)
        return model, roc, pr

    @staticmethod
    def train_catboost(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_weights: np.ndarray = None
    ) -> Tuple[Any, float, float]:
        from catboost import CatBoostClassifier
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        model = CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.04,
            scale_pos_weight=scale_pos_weight,
            eval_metric="AUC",
            random_seed=42,
            verbose=0
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        test_probs = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, test_probs)
        pr = average_precision_score(y_test, test_probs)
        return model, roc, pr

