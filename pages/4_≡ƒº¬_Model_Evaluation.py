"""Confusion matrices, reports, grouped CV, and feature interpretation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Model Evaluation")
root = Path(__file__).resolve().parents[1]
metrics_path = root / "models" / "metrics.json"
if not metrics_path.exists():
    st.info("No evaluation artifacts yet. Run `python -m src.training` first.")
    st.stop()
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
names = list(metrics["models"])
selected = st.selectbox("Classifier", names, index=names.index(metrics["best_model"]))
result = metrics["models"][selected]
cols = st.columns(4)
for col, label, key in zip(cols, ["Test accuracy", "Macro precision", "Macro recall", "Macro F1"],
                           ["accuracy", "precision_macro", "recall_macro", "f1_macro"]):
    col.metric(label, f"{result[key]:.3f}")
labels = result["labels"]
matrix = np.asarray(result["confusion_matrix"])
st.plotly_chart(px.imshow(matrix, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                          labels={"x": "Predicted stage", "y": "True stage", "color": "Epochs"},
                          title=f"{selected} · held-out subject confusion matrix"), width="stretch")
st.subheader("Per-stage classification report")
st.dataframe(pd.DataFrame(result["classification_report"]).T, width="stretch")
st.subheader("Grouped cross-validation and subject split")
st.write(f"Training-only stratified grouped folds: {result['cv_folds']} · training epochs: {result['train_records']:,} · "
         f"test epochs: {result['test_records']:,}")
st.write(f"Train macro F1: {result['train_f1_macro']:.3f} · test macro F1: {result['f1_macro']:.3f} · "
         f"train-test gap: {result['generalization_gap_f1']:.3f}")
st.write(f"Training subjects: {', '.join(result['train_subjects'])}")
st.write(f"Held-out subjects: {', '.join(result['test_subjects'])}")
cv_frame = pd.DataFrame({"Fold": range(1, len(result["cross_validation_f1_macro"]) + 1),
                         "Validation macro F1": result["cross_validation_f1_macro"]})
st.plotly_chart(px.bar(cv_frame, x="Fold", y="Validation macro F1", title="Training-only grouped CV scores",
                       range_y=[0, 1]), width="stretch")
st.write("Best parameters:", result["best_params"])
importance = metrics.get("feature_importance", {}).get(selected)
if importance:
    frame = pd.DataFrame({"Feature": list(importance), "Importance": list(importance.values())})
    st.plotly_chart(px.bar(frame.nlargest(15, "Importance").sort_values("Importance"), x="Importance",
                           y="Feature", orientation="h", title=f"{selected} feature contribution"),
                    width="stretch")
else:
    st.caption("No native coefficient/impurity importance is available for this estimator.")
st.caption("The subject IDs above are used only to verify grouped splits; they are never model inputs. "
           "Predictions are experimental, not medical diagnoses.")
