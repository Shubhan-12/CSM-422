"""Explore the supplied EEG epochs, labels, and extracted signal features."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loading import LABEL_MAPPING, STAGE_ORDER, list_recordings, load_recording, subject_id_from_path
from src.training import build_feature_dataset

ROOT = Path(__file__).resolve().parents[1]
st.title("Dataset Analysis")
st.caption("Preprocessed Sleep-EDF-20 EEG, one channel (Fpz-Cz), 30 seconds per epoch, sampled at 100 Hz.")

@st.cache_data(show_spinner="Extracting features from the local EEG epochs…")
def cached_features():
    X, y, groups, recording_ids = build_feature_dataset(ROOT / "archive (11)")
    return X, y, groups, recording_ids

try:
    files = list_recordings(ROOT / "archive (11)")
    counts = pd.Series(0, index=STAGE_ORDER, dtype="int64")
    for path in files:
        _signal, codes, _fs, _channel = load_recording(path)
        for code, count in pd.Series(codes).value_counts().items():
            counts[LABEL_MAPPING[int(code)]] += int(count)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(counts.rename_axis("Stage").reset_index(name="Epochs"), x="Stage", y="Epochs",
                                color="Stage", category_orders={"Stage": STAGE_ORDER},
                                title="Sleep-stage distribution"), width="stretch")
    with right:
        recording = st.selectbox("Recording", files, format_func=lambda p: p.name)
        signal, labels, fs, channel = load_recording(recording)
        epoch_index = st.slider("Epoch index", 0, len(labels) - 1, min(100, len(labels)-1))
        seconds = pd.Series(range(signal.shape[1])) / fs
        chart = pd.DataFrame({"Seconds": seconds, "EEG amplitude": signal[epoch_index]})
        st.plotly_chart(px.line(chart, x="Seconds", y="EEG amplitude", title=f"{channel} · epoch {epoch_index} · {LABEL_MAPPING[int(labels[epoch_index])]}") , width="stretch")
    st.write(f"**{len(files)} recordings · {len({subject_id_from_path(f) for f in files})} subjects · "
             f"{int(counts.sum()):,} labeled epochs.** The provided files contain only stage codes 0–4; "
             "there are no movement or unknown-label epochs to exclude.")

    X, y, groups, recording_ids = cached_features()
    st.subheader("EEG feature distributions")
    feature = st.selectbox("Feature", list(X.columns), index=list(X.columns).index("delta_relative_power"))
    plot_data = pd.DataFrame({feature: X[feature], "Sleep stage": y})
    st.plotly_chart(px.box(plot_data, x="Sleep stage", y=feature, color="Sleep stage",
                           category_orders={"Sleep stage": STAGE_ORDER}, points=False), width="stretch")
    st.subheader("Feature summary by stage")
    summary = X.assign(stage=y).groupby("stage").mean(numeric_only=True).reindex(STAGE_ORDER)
    st.dataframe(summary, width="stretch")
    st.caption("Features use time-domain statistics, Hjorth parameters, spectral entropy, dominant frequency, "
               "and absolute/relative delta, theta, alpha, sigma, and beta power. IDs are retained for grouping only.")
except (OSError, ValueError, KeyError, RuntimeError) as exc:
    st.error(f"Could not analyze the local Sleep-EDF-20 files: {exc}")
