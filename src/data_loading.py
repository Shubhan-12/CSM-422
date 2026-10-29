"""Inventory and load the local preprocessed Sleep-EDF-20 NPZ recordings."""

from collections import Counter
from pathlib import Path
import re

import numpy as np

DATASET_DIRNAME = "archive (11)"
# Dataset export encoding confirmed against the source archive's class counts
# and Sleep-EDF labels: W, 1, 2, 3/4 merged as N3, and R.
LABEL_MAPPING = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]
EXPECTED_KEYS = {"x", "y", "fs", "ch_label"}


def dataset_root(project_root=None):
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    path = root / DATASET_DIRNAME
    if not path.is_dir():
        raise FileNotFoundError(f"Sleep-EDF-20 folder not found: {path}")
    return path


def subject_id_from_path(path):
    match = re.match(r"(SC\d{3})", Path(path).stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot derive Sleep-EDF subject ID from filename: {Path(path).name}")
    return match.group(1).upper()


def list_recordings(data_dir=None):
    """Find NPZ files recursively; do not download or modify source files."""
    root = Path(data_dir) if data_dir is not None else dataset_root()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz recordings found under {root}")
    return files


def load_recording(path):
    """Read EEG epochs and integer targets safely without loading object metadata."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = EXPECTED_KEYS.difference(archive.files)
            if missing:
                raise ValueError(f"{path.name} is missing expected keys: {sorted(missing)}")
            x = np.asarray(archive["x"])
            y = np.asarray(archive["y"])
            fs = float(archive["fs"])
            channel = str(archive["ch_label"].item())
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not load NPZ recording {path}: {exc}") from exc
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y):
        raise ValueError(f"Unexpected EEG/label dimensions in {path.name}: x={x.shape}, y={y.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"EEG signal contains NaN or infinite values in {path.name}")
    unknown = sorted(set(np.unique(y).tolist()).difference(LABEL_MAPPING))
    if unknown:
        raise ValueError(f"Unrecognized stage codes in {path.name}: {unknown}")
    return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False), fs, channel


def inspect_dataset(data_dir=None):
    """Return measured archive and target statistics without feature extraction."""
    root = Path(data_dir) if data_dir is not None else dataset_root()
    files = list_recordings(root)
    counts = Counter()
    recordings_by_subject = Counter()
    file_formats = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())
    sample_shapes, sample_rates, channels = Counter(), Counter(), Counter()
    total_bytes = 0
    invalid = 0
    epoch_count = 0
    for path in files:
        x, y, fs, channel = load_recording(path)
        epoch_count += len(y)
        counts.update(map(int, y))
        recordings_by_subject[subject_id_from_path(path)] += 1
        sample_shapes[str(x.shape[1:])] += 1
        sample_rates[str(fs)] += 1
        channels[channel] += 1
        total_bytes += path.stat().st_size
        invalid += int(np.count_nonzero(~np.isfinite(x)))
    return {
        "root": str(root), "files": files, "file_count": len(files),
        "file_formats": dict(file_formats), "subjects": sorted(recordings_by_subject),
        "subject_count": len(recordings_by_subject), "recordings_by_subject": dict(recordings_by_subject),
        "epoch_count": epoch_count, "stage_counts_codes": dict(counts),
        "stage_counts": {LABEL_MAPPING[code]: int(counts.get(code, 0)) for code in LABEL_MAPPING},
        "sample_shapes": dict(sample_shapes), "sample_rates_hz": dict(sample_rates),
        "channels": dict(channels), "bytes": total_bytes, "invalid_signal_values": invalid,
    }
