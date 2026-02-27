import numpy as np


def confusion_matrix_scratch(y_true, y_pred):
    """Compute confusion matrix values from scratch."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def evaluate_classification(y_true, y_pred):
    """Compute and print full classification metrics from scratch."""
    tp, tn, fp, fn = confusion_matrix_scratch(y_true, y_pred)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0          # Sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0     # True Negative Rate
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Confusion Matrix: TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
