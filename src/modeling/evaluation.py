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


def evaluate_regression(y_true, y_pred):
    """Compute and print regression metrics."""
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    print(f"  MSE:   {mse:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  R2:    {r2:.4f}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def compare_classification(scratch_metrics, lib_metrics, model_name="Model"):
    """Print side-by-side comparison table for classification metrics."""
    print("\n" + "=" * 60)
    print(f"  Comparison: {model_name}")
    print("=" * 60)
    print(f"  {'Metric':<15} {'Scratch':>12} {'Lib (sklearn)':>14} {'Diff':>10}")
    print("  " + "-" * 53)
    for key in ["accuracy", "precision", "recall", "specificity", "f1"]:
        s = scratch_metrics.get(key, 0)
        lib_val = lib_metrics.get(key, 0)
        diff = lib_val - s
        sign = "+" if diff >= 0 else ""
        print(f"  {key:<15} {s:>12.4f} {lib_val:>14.4f} {sign}{diff:>9.4f}")
    print("=" * 60)


def compare_regression(scratch_metrics, lib_metrics, model_name="Model"):
    """Print side-by-side comparison table for regression metrics."""
    print("\n" + "=" * 60)
    print(f"  Comparison: {model_name}")
    print("=" * 60)
    print(f"  {'Metric':<15} {'Scratch':>12} {'Lib (sklearn)':>14} {'Diff':>10}")
    print("  " + "-" * 53)
    for key in ["mse", "rmse", "mae", "r2"]:
        s = scratch_metrics.get(key, 0)
        lib_val = lib_metrics.get(key, 0)
        diff = lib_val - s
        sign = "+" if diff >= 0 else ""
        print(f"  {key:<15} {s:>12.4f} {lib_val:>14.4f} {sign}{diff:>9.4f}")
    print("=" * 60)
