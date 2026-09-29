import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)

class CalibrationReport:
    """Computes clinical decision metrics and threshold calibration."""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))

        # Youden's J threshold (Sensitivity + Specificity - 1)
        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        opt_youden_thresh = float(roc_thresh[best_j_idx])

        # F1-maximizing threshold
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        opt_f1_thresh = float(pr_thresh[min(best_f1_idx, len(pr_thresh) - 1)])

        # Predictions at Youden's threshold (Clinical screening)
        preds_youden = (y_prob >= opt_youden_thresh).astype(int)
        preds_f1 = (y_prob >= opt_f1_thresh).astype(int)

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "youden_threshold": opt_youden_thresh,
            "f1_threshold": opt_f1_thresh,
            "report_youden": classification_report(y_true, preds_youden, target_names=["No Diabetes", "Diagnosed Diabetes"], output_dict=True),
            "cm_youden": confusion_matrix(y_true, preds_youden).tolist(),
            "report_f1": classification_report(y_true, preds_f1, target_names=["No Diabetes", "Diagnosed Diabetes"], output_dict=True),
            "cm_f1": confusion_matrix(y_true, preds_f1).tolist(),
        }

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    return CalibrationReport.compute(y_true, y_prob)
