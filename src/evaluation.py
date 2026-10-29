"""Reusable fixed-label metrics and subject-grouped cross-validation helpers."""

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score


def evaluate_predictions(y_true, y_pred, labels):
    """Return consistent five-class held-out metrics in the requested label order."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels,
                                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def grouped_cross_validation(estimator, X, y, groups, labels, *, folds=4, random_state=43):
    """Score an unfitted pipeline with stratified grouped folds, never splitting subjects."""
    unique_groups = len(np.unique(groups))
    n_splits = min(int(folds), unique_groups)
    if n_splits < 2:
        raise ValueError("At least two subject groups are required for grouped cross-validation")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, labels=list(labels), average="macro", zero_division=0)
    scores = cross_val_score(estimator, X, y, groups=groups, cv=splitter, scoring=scorer, n_jobs=1)
    return {"folds": n_splits, "macro_f1_scores": scores.tolist(),
            "mean_macro_f1": float(scores.mean()), "std_macro_f1": float(scores.std())}
