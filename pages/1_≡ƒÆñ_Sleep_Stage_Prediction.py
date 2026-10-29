"""Predict Sleep-EDF sleep stages from compatible preprocessed EEG NPZ files."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.prediction import load_model, predict_npz_bytes

ROOT = Path(__file__).resolve().parents[1]
st.title("Sleep-Stage Prediction")
st.write("Upload a compatible `.npz` file with `x` (30-second EEG epochs), `fs` (100 Hz), and `ch_label` arrays. "
         "The provided dataset uses the Fpz-Cz channel and 3,000 samples per epoch.")
st.warning("Predictions are experimental and are not medical diagnoses.")
model_path = ROOT / "models" / "sleep_stage_model.joblib"
if not model_path.exists():
    st.error("Sleep-stage artifacts are missing. Train with `python -m src.training` first.")
    st.stop()

try:
    bundle = load_model(model_path)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not load the Sleep-EDF model: {exc}")
    st.stop()

uploaded = st.file_uploader("Preprocessed EEG recording (.npz)", type=["npz"])
if uploaded is not None:
    if uploaded.size > 100 * 1024 * 1024:
        st.error("File is larger than 100 MB. Upload one compatible recording at a time.")
    elif st.button("Predict sleep stages", type="primary"):
        try:
            result = predict_npz_bytes(uploaded.getvalue(), uploaded.name, model_path=model_path)
            stages = result["predictions"]
            st.success(f"Predicted {len(stages):,} epochs from channel **{result['channel']}**.")
            st.dataframe(pd.DataFrame({"Epoch": range(len(stages)), "Predicted stage": stages}),
                         width="stretch", hide_index=True)
            counts = (pd.Series(stages, name="Sleep stage").value_counts()
                      .reindex(bundle["classes"], fill_value=0).rename_axis("Sleep stage")
                      .reset_index(name="Epochs"))
            st.plotly_chart(px.bar(counts, x="Sleep stage", y="Epochs", color="Sleep stage",
                                   category_orders={"Sleep stage": bundle["classes"]},
                                   title="Predicted stage counts"), width="stretch")
            if result["probabilities"] is None:
                st.caption("This selected SVM returns class decisions but is not calibrated to provide class probabilities.")
            else:
                st.subheader("First epoch class probabilities")
                probability_table = pd.DataFrame({
                    "Sleep stage": bundle["model"].classes_,
                    "Probability": result["probabilities"][0],
                })
                st.bar_chart(probability_table.set_index("Sleep stage"), y="Probability")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Prediction failed: {exc}")
