"""Train subject-independent sleep-stage models from local Sleep-EDF NPZ files."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loading import LABEL_MAPPING, STAGE_ORDER, dataset_root, list_recordings, load_recording, subject_id_from_path
from src.signal_processing import FEATURE_COLUMNS, extract_epoch_features

TARGET = "sleep_stage"


def build_feature_dataset(data_dir=None):
    """Load the local NPZ files and derive features for every labelled epoch."""
    records, targets, subjects, recordings = [], [], [], []
    inventory = list_recordings(data_dir)
    for path in inventory:
        eeg, codes, fs, _channel = load_recording(path)
        features = extract_epoch_features(eeg, fs)
        records.append(features)
        targets.append(pd.Series([LABEL_MAPPING[int(code)] for code in codes], name=TARGET))
        subjects.extend([subject_id_from_path(path)] * len(codes))
        recordings.extend([path.stem] * len(codes))
    X = pd.concat(records, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)
    groups = pd.Series(subjects, name="subject_id")
    recording_ids = pd.Series(recordings, name="recording_id")
    if list(X.columns) != FEATURE_COLUMNS:
        raise RuntimeError("Feature order differs from the declared feature schema")
    if not set(y.unique()).issubset(STAGE_ORDER):
        raise ValueError("Unexpected target labels found after Sleep-EDF label mapping")
    return X, y, groups, recording_ids


def build_pipeline(classifier):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def model_searches(random_state=42):
    """Grids kept bounded for the 42k-epoch data; LinearSVC keeps SVM tractable."""
    return {
        "Logistic Regression": (
            build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            {"classifier__C": [0.1, 1.0]},
        ),
        "Decision Tree": (
            build_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=random_state)),
            {"classifier__max_depth": [12, None], "classifier__min_samples_leaf": [1, 5]},
        ),
        "Random Forest": (
            build_pipeline(RandomForestClassifier(class_weight="balanced", random_state=random_state,
                                                  n_jobs=1, n_estimators=120)),
            {"classifier__max_depth": [20, None], "classifier__min_samples_leaf": [1, 3]},
        ),
        "SVM": (
            build_pipeline(LinearSVC(class_weight="balanced", random_state=random_state, dual="auto", max_iter=5000)),
            {"classifier__C": [0.1, 1.0]},
        ),
    }


def _subject_holdout(X, y, groups, random_state):
    """Use grouped stratification; reject a split missing any class on either side."""
    split_count = min(5, groups.nunique())
    splitter = StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=random_state)
    all_labels = set(STAGE_ORDER)
    for train_idx, test_idx in splitter.split(X, y, groups):
        if set(y.iloc[train_idx]) == all_labels and set(y.iloc[test_idx]) == all_labels:
            return train_idx, test_idx, split_count
    raise ValueError("Could not form a subject-disjoint train/test split containing all five stages. "
                     "More labelled subjects per stage are needed for this evaluation strategy.")


def train_models(data_dir=None, output_dir="models", random_state=42):
    """Tune on subject-grouped training folds and evaluate once on held-out subjects."""
    data_dir = Path(data_dir) if data_dir is not None else dataset_root()
    X, y, groups, recording_ids = build_feature_dataset(data_dir)
    train_idx, test_idx, _outer_folds = _subject_holdout(X, y, groups, random_state)
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    split_count = min(4, groups_train.nunique())
    cv = StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=random_state + 1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model_pipelines, results = {}, {}
    for name, (pipeline, params) in model_searches(random_state).items():
        start = time.perf_counter()
        fixed_macro_f1 = make_scorer(f1_score, labels=STAGE_ORDER, average="macro", zero_division=0)
        search = GridSearchCV(pipeline, params, scoring=fixed_macro_f1, cv=cv, n_jobs=2,
                              refit=True, return_train_score=True, error_score="raise")
        search.fit(X_train, y_train, groups=groups_train)
        elapsed = time.perf_counter() - start
        estimator = search.best_estimator_
        train_pred, test_pred = estimator.predict(X_train), estimator.predict(X_test)
        best = search.best_index_
        cv_scores = [float(search.cv_results_[f"split{i}_test_score"][best]) for i in range(split_count)]
        model_pipelines[name] = estimator
        results[name] = {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "precision_macro": float(precision_score(y_test, test_pred, labels=STAGE_ORDER, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, test_pred, labels=STAGE_ORDER, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, test_pred, labels=STAGE_ORDER, average="macro", zero_division=0)),
            "train_f1_macro": float(f1_score(y_train, train_pred, labels=STAGE_ORDER, average="macro", zero_division=0)),
            "generalization_gap_f1": float(f1_score(y_train, train_pred, labels=STAGE_ORDER, average="macro", zero_division=0) - f1_score(y_test, test_pred, labels=STAGE_ORDER, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, test_pred, labels=STAGE_ORDER).tolist(),
            "labels": STAGE_ORDER,
            "classification_report": classification_report(y_test, test_pred, labels=STAGE_ORDER, output_dict=True, zero_division=0),
            "cross_validation_f1_macro": cv_scores,
            "cross_validation_f1_macro_mean": float(search.best_score_),
            "cross_validation_f1_macro_std": float(search.cv_results_["std_test_score"][best]),
            "best_params": search.best_params_, "training_time_seconds": float(elapsed),
            "cv_folds": split_count, "train_records": int(len(train_idx)), "test_records": int(len(test_idx)),
            "train_subjects": sorted(groups.iloc[train_idx].unique().tolist()),
            "test_subjects": sorted(groups.iloc[test_idx].unique().tolist()),
            "test_recordings": sorted(recording_ids.iloc[test_idx].unique().tolist()),
        }
        joblib.dump(estimator, output / f"{name.lower().replace(' ', '_')}.joblib")

    best_model = max(results, key=lambda name: results[name]["cross_validation_f1_macro_mean"])
    bundle = {
        "model": model_pipelines[best_model], "model_name": best_model, "models": model_pipelines,
        "feature_columns": FEATURE_COLUMNS, "classes": STAGE_ORDER, "label_mapping": LABEL_MAPPING,
        "sampling_rate_hz": 100.0, "epoch_samples": 3000, "epoch_seconds": 30,
        "dataset": "Sleep-EDF-20 preprocessed local NPZ",
    }
    joblib.dump(bundle, output / "sleep_stage_model.joblib")

    feature_importance = {}
    for name, estimator in model_pipelines.items():
        clf = estimator.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            scores = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            scores = np.mean(np.abs(clf.coef_), axis=0)
        else:
            continue
        feature_importance[name] = {feature: float(score) for feature, score in zip(FEATURE_COLUMNS, scores)}

    all_codes = {int(k): str(v) for k, v in LABEL_MAPPING.items()}
    metrics = {
        "dataset": "Sleep-EDF-20 preprocessed local NPZ", "dataset_directory": str(data_dir),
        "dataset_records": int(len(X)), "recording_count": int(recording_ids.nunique()),
        "subject_count": int(groups.nunique()), "stage_order": STAGE_ORDER,
        "label_mapping": all_codes, "class_counts": {stage: int((y == stage).sum()) for stage in STAGE_ORDER},
        "feature_columns": FEATURE_COLUMNS, "excluded_columns": ["subject_id", "recording_id"],
        "selection_metric": "Mean StratifiedGroupKFold macro F1 on training subjects",
        "split_strategy": "Subject-disjoint StratifiedGroupKFold holdout; training-only grouped CV for tuning",
        "best_model": best_model, "models": results, "feature_importance": feature_importance,
        "excluded_unscored_epochs": 0, "subject_overlap": sorted(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    comparison = pd.DataFrame.from_dict(results, orient="index").rename_axis("Model").reset_index()
    comparison.to_csv(output / "model_comparison.csv", index=False)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_models(), indent=2))
