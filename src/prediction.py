"""Load the Sleep-EDF model and predict from compatible preprocessed EEG NPZ."""

from pathlib import Path
from io import BytesIO

import joblib
import numpy as np

from src.preprocessing import prepare_signal_features


def load_model(path="models/sleep_stage_model.joblib"):
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Sleep-stage model not found: {model_path}. Run `python -m src.training` first.")
    bundle = joblib.load(model_path)
    if bundle.get("dataset") != "Sleep-EDF-20 preprocessed local NPZ":
        raise ValueError("The saved model is not a Sleep-EDF-20 model. Retrain with `python -m src.training`.")
    return bundle


def predict_epochs(epochs, sampling_rate, bundle=None, model_path="models/sleep_stage_model.joblib"):
    """Predict labels for pre-segmented 30-second EEG epochs with training features."""
    model_bundle = bundle if bundle is not None else load_model(model_path)
    signals = np.asarray(epochs)
    if signals.ndim == 3 and signals.shape[-1] == 1:
        signals = signals[..., 0]
    if signals.ndim == 1:
        signals = signals[None, :]
    if signals.ndim != 2 or signals.shape[1] != model_bundle["epoch_samples"]:
        raise ValueError(f"Expected EEG shaped (epochs, {model_bundle['epoch_samples']}) for "
                         f"{model_bundle['epoch_seconds']}-second epochs; received {signals.shape}.")
    if float(sampling_rate) != float(model_bundle["sampling_rate_hz"]):
        raise ValueError(f"Expected sampling rate {model_bundle['sampling_rate_hz']} Hz; received {sampling_rate} Hz.")
    features = prepare_signal_features(signals, sampling_rate)
    predictions = model_bundle["model"].predict(features)
    probabilities = None
    if hasattr(model_bundle["model"], "predict_proba"):
        probabilities = model_bundle["model"].predict_proba(features)
    return {"predictions": predictions.tolist(), "probabilities": probabilities, "features": features}


def predict_npz_bytes(file_bytes, filename="uploaded.npz", model_path="models/sleep_stage_model.joblib"):
    """Validate uploaded NPZ without unpickling metadata and predict all epochs."""
    if Path(filename).suffix.lower() != ".npz":
        raise ValueError("Unsupported recording format. Upload a preprocessed .npz Sleep-EDF epoch file.")
    try:
        with np.load(BytesIO(file_bytes), allow_pickle=False) as archive:
            missing = {"x", "fs", "ch_label"}.difference(archive.files)
            if missing:
                raise ValueError(f"NPZ is missing required arrays: {', '.join(sorted(missing))}")
            signals = np.asarray(archive["x"])
            sampling_rate = float(archive["fs"])
            channel = str(archive["ch_label"].item())
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not read compatible NPZ recording: {exc}") from exc
    bundle = load_model(model_path)
    result = predict_epochs(signals, sampling_rate, bundle=bundle)
    result["channel"] = channel
    return result
