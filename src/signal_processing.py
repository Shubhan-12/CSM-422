"""Turn fixed 30-second single-channel EEG epochs into interpretable features."""

import numpy as np
import pandas as pd
from scipy.signal import welch

SAMPLE_SECONDS = 30
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}
FEATURE_COLUMNS = [
    "signal_mean", "signal_std", "signal_rms", "hjorth_mobility", "hjorth_complexity",
    "spectral_entropy", "dominant_frequency",
    *[f"{band}_{kind}_power" for band in BANDS for kind in ("absolute", "relative")],
]


def extract_epoch_features(epochs, sampling_rate):
    """Vectorized features for an (epochs, samples) array, computed per epoch."""
    signals = np.asarray(epochs, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals[None, :]
    if signals.ndim != 2 or signals.shape[1] < 16:
        raise ValueError("EEG must have shape (epochs, samples) with at least 16 samples per epoch")
    if not np.isfinite(signals).all() or sampling_rate <= 0:
        raise ValueError("EEG values must be finite and sampling rate must be positive")

    centered = signals - signals.mean(axis=1, keepdims=True)
    variance = np.mean(centered ** 2, axis=1)
    first = np.diff(centered, axis=1)
    second = np.diff(first, axis=1)
    first_var = np.mean(first ** 2, axis=1)
    second_var = np.mean(second ** 2, axis=1)
    mobility = np.sqrt(np.divide(first_var, variance, out=np.zeros_like(variance), where=variance > 0))
    d_mobility = np.sqrt(np.divide(second_var, first_var, out=np.zeros_like(first_var), where=first_var > 0))
    complexity = np.divide(d_mobility, mobility, out=np.zeros_like(mobility), where=mobility > 0)

    frequencies, psd = welch(centered, fs=float(sampling_rate), nperseg=min(signals.shape[1], 1024), axis=1)
    total_power = np.trapezoid(psd, frequencies, axis=1)
    safe_total = np.where(total_power > 0, total_power, 1.0)
    density = psd / np.maximum(psd.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    entropy = -np.sum(density * np.log2(np.maximum(density, np.finfo(float).tiny)), axis=1)
    dominant = frequencies[np.argmax(psd, axis=1)]

    result = {
        "signal_mean": signals.mean(axis=1),
        "signal_std": signals.std(axis=1),
        "signal_rms": np.sqrt(np.mean(signals ** 2, axis=1)),
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "spectral_entropy": entropy,
        "dominant_frequency": dominant,
    }
    for band, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        power = np.trapezoid(psd[:, mask], frequencies[mask], axis=1) if mask.sum() > 1 else np.zeros(len(signals))
        result[f"{band}_absolute_power"] = power
        result[f"{band}_relative_power"] = power / safe_total
    return pd.DataFrame(result, columns=FEATURE_COLUMNS)


def features_from_uploaded_epochs(epochs, sampling_rate):
    """Validate uploaded EEG windows and use the identical training extractor."""
    return extract_epoch_features(epochs, sampling_rate)
