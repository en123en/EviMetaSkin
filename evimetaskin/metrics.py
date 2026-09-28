"""Evaluation metrics and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, task: str, y_score=None):
    average = "binary" if task == "binary" else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if task == "binary":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update(
            {
                "specificity": float(tn / (tn + fp + 1e-8)),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
        if y_score is not None and len(set(y_true)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def binary_preds_from_scores(scores, threshold: float):
    return (np.asarray(scores) >= float(threshold)).astype(int).tolist()


def select_binary_threshold(y_true, y_score, mode: str, fixed_threshold: float = 0.5, min_recall: float = 0.85):
    if mode == "fixed":
        threshold = float(fixed_threshold)
        return threshold, compute_metrics(y_true, binary_preds_from_scores(y_score, threshold), "binary", y_score)

    best_threshold, best_metrics, best_value = 0.5, None, -np.inf
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = compute_metrics(y_true, binary_preds_from_scores(y_score, threshold), "binary", y_score)
        if mode == "f1":
            value = metrics["f1"]
        elif mode == "balanced_accuracy":
            value = metrics["balanced_accuracy"]
        elif mode == "recall_constraint":
            if metrics["recall"] < float(min_recall):
                continue
            value = metrics["specificity"]
        else:
            raise ValueError(f"Unsupported threshold selection mode: {mode}")
        if value > best_value:
            best_threshold, best_metrics, best_value = float(threshold), metrics, value

    if best_metrics is None:
        best_metrics = compute_metrics(y_true, binary_preds_from_scores(y_score, 0.5), "binary", y_score)
    return best_threshold, best_metrics
