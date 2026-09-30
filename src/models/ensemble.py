import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

class TabularGraphEnsemble:
    """
    Combines gradient-boosted tabular trees (LightGBM/XGBoost) with 
    Graph Attention Networks (GATv2) via probability blending, meta-learning stacking,
    and latent graph embedding fusion.
    """

    @staticmethod
    def find_optimal_blend(
        val_probs_a: np.ndarray,
        val_probs_b: np.ndarray,
        y_val: np.ndarray,
        metric: str = "roc_auc",
        steps: int = 101
    ) -> Tuple[float, float]:
        """
        Finds the optimal weight beta in [0, 1] on the validation set:
        pred = beta * probs_a + (1 - beta) * probs_b
        """
        best_score = -1.0
        best_beta = 0.5

        for beta in np.linspace(0.0, 1.0, steps):
            blend = beta * val_probs_a + (1.0 - beta) * val_probs_b
            if metric == "pr_auc":
                score = average_precision_score(y_val, blend)
            else:
                score = roc_auc_score(y_val, blend)

            if score > best_score:
                best_score = score
                best_beta = float(beta)

        return best_beta, best_score

    @staticmethod
    def find_optimal_multimodel_blend(
        val_probs_matrix: np.ndarray,
        y_val: np.ndarray,
        metric: str = "roc_auc"
    ) -> np.ndarray:
        """
        Optimizes non-negative blending weights summing to 1 across M models
        using SLSQP constrained optimization on the validation set.
        """
        from scipy.optimize import minimize
        n_models = val_probs_matrix.shape[1]
        init_weights = np.ones(n_models) / n_models

        def objective(weights):
            w = np.maximum(weights, 0)
            if np.sum(w) == 0:
                return 0.0
            w = w / np.sum(w)
            pred = np.dot(val_probs_matrix, w)
            if metric == "pr_auc":
                return -average_precision_score(y_val, pred)
            return -roc_auc_score(y_val, pred)

        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        res = minimize(objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        best_w = np.maximum(res.x, 0)
        return best_w / np.sum(best_w)

    @staticmethod
    def train_meta_learner(
        val_features: np.ndarray,
        y_val: np.ndarray,
        test_features: np.ndarray,
        y_test: np.ndarray,
        C: float = 1.0
    ) -> Tuple[LogisticRegression, np.ndarray, float, float]:
        """
        Trains an L2-regularized Logistic Regression meta-learner on validation probabilities.
        """
        meta = LogisticRegression(C=C, class_weight="balanced", random_state=42, max_iter=200)
        meta.fit(val_features, y_val)
        meta_test_probs = meta.predict_proba(test_features)[:, 1]
        roc = roc_auc_score(y_test, meta_test_probs)
        pr = average_precision_score(y_test, meta_test_probs)
        return meta, meta_test_probs, roc, pr



    @staticmethod
    def train_gnn_stacked_lightgbm(
        X_train: np.ndarray,
        h_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        h_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[lgb.LGBMClassifier, np.ndarray, float, float]:
        """
        Concatenates tabular features X with 64-dim latent GNN embeddings h,
        then trains LightGBM on the unified representation [X || h].
        """
        X_train_stacked = np.hstack([X_train, h_train])
        X_test_stacked = np.hstack([X_test, h_test])

        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train_stacked, y_train)
        test_probs = model.predict_proba(X_test_stacked)[:, 1]
        roc = roc_auc_score(y_test, test_probs)
        pr = average_precision_score(y_test, test_probs)
        return model, test_probs, roc, pr
