"""Sleep-EDF label and epoch validation helpers."""

import numpy as np
import pandas as pd

from src.data_loading import LABEL_MAPPING, STAGE_ORDER
from src.signal_processing import FEATURE_COLUMNS, extract_epoch_features


def map_stage_codes(codes):
    """Map dataset integer stage codes to the five named sleep stages."""
    values = np.asarray(codes)
    unknown = sorted(set(np.unique(values).tolist()).difference(LABEL_MAPPING))
    if unknown:
        raise ValueError(f"Unknown or unscored stage codes cannot be mapped: {unknown}")
    return pd.Series([LABEL_MAPPING[int(code)] for code in values], name="sleep_stage")


def prepare_signal_features(epochs, sampling_rate):
    """Apply the shared deterministic signal feature extraction and schema check."""
    features = extract_epoch_features(epochs, sampling_rate)
    if list(features.columns) != FEATURE_COLUMNS or features.empty:
        raise ValueError("EEG preprocessing did not produce the expected non-empty feature set")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Derived EEG features contain NaN or infinite values")
    return features


def validate_stage_labels(labels):
    unexpected = sorted(set(labels).difference(STAGE_ORDER))
    if unexpected:
        raise ValueError(f"Unsupported sleep-stage labels: {unexpected}")
    return labels
