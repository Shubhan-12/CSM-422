"""Compare the four subject-independent Sleep-EDF models."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Model Comparison")
path = Path(__file__).resolve().parents[1] / "models" / "metrics.json"
if not path.exists():
    st.info("No trained models yet. Run `python -m src.training` from the project root.")
    st.stop()
metrics = json.loads(path.read_text(encoding="utf-8"))
st.caption(f"Target: five sleep stages · {metrics['split_strategy']} · selected by training-only grouped CV macro F1.")
frame = pd.DataFrame.from_dict(metrics["models"], orient="index").rename_axis("Model").reset_index()
st.dataframe(frame[["Model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "cross_validation_f1_macro_mean", "cross_validation_f1_macro_std", "training_time_seconds"]]
             .rename(columns={"accuracy": "Test accuracy", "precision_macro": "Test macro precision",
                              "recall_macro": "Test macro recall", "f1_macro": "Test macro F1",
                              "cross_validation_f1_macro_mean": "Train CV macro F1 mean",
                              "cross_validation_f1_macro_std": "Train CV macro F1 SD",
                              "training_time_seconds": "Grid search seconds"}),
             width="stretch", hide_index=True)
long = frame.melt(id_vars="Model", value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                  var_name="Test metric", value_name="Score")
st.plotly_chart(px.bar(long, x="Model", y="Score", color="Test metric", barmode="group",
                       title="Held-out subject test metrics", range_y=[0, 1]), width="stretch")
cv = frame[["Model", "cross_validation_f1_macro_mean", "cross_validation_f1_macro_std"]].rename(
    columns={"cross_validation_f1_macro_mean": "Mean CV macro F1", "cross_validation_f1_macro_std": "SD"})
st.plotly_chart(px.bar(cv, x="Model", y="Mean CV macro F1", error_y="SD",
                       title="Training-only subject-grouped CV (selection metric)", range_y=[0, 1]), width="stretch")
st.write(f"**Selected model: {metrics['best_model']}** · mean grouped CV macro F1 "
         f"{metrics['models'][metrics['best_model']]['cross_validation_f1_macro_mean']:.3f}.")
