"""Sleep-EDF-20 sleep-stage dashboard home page."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loading import STAGE_ORDER, dataset_root, inspect_dataset

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Sleep Signal Lab", page_icon="🌙", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#172b3a; --muted:#607487; --teal:#087e83; --mint:#d9f1eb; --paper:#f5f8f7; }
.stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:var(--ink); }
[data-testid="stSidebar"] { background:#edf4f2; border-right:1px solid #dce8e5; }
[data-testid="stMetric"] { background:white; padding:18px 20px; border:1px solid #e3ece9; border-radius:16px; box-shadow:0 5px 18px #173c3310; }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:var(--teal); font-family:'Manrope',sans-serif; }
.hero { background:linear-gradient(120deg,#103b45 0%,#087e83 58%,#35a68c 100%); padding:36px 40px; border-radius:24px; color:white; margin:8px 0 24px; box-shadow:0 16px 36px #0d596122; }
.hero h1 { color:white !important; font-size:2.65rem; margin:8px 0 12px; }
.hero p { color:#e2f3ef; font-size:1.08rem; max-width:780px; line-height:1.7; margin:0; }
.eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.75rem; font-weight:700; color:#b7e9dc; }
.step-card { background:white; border:1px solid #e3ece9; border-radius:16px; padding:20px; min-height:142px; }
.step-num { color:var(--teal); font-size:.78rem; font-weight:700; letter-spacing:.12em; }
.step-card h3 { font-size:1.12rem; margin:8px 0; }
.step-card p { color:var(--muted); font-size:.92rem; line-height:1.5; margin:0; }
div[data-testid="stAlert"] { border-radius:14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<section class="hero">
  <div class="eyebrow">Sleep Signal Lab · machine learning project</div>
  <h1>Reading sleep, one EEG epoch at a time.</h1>
  <p>Explore how signal-processing features and machine-learning models classify five sleep stages from single-channel EEG. Start with the dataset, review model performance, or try a compatible preprocessed recording.</p>
</section>
""", unsafe_allow_html=True)
st.info("Research and learning prototype. Predictions are experimental and are not a medical diagnosis.", icon="ⓘ")

try:
    inventory = inspect_dataset(dataset_root(ROOT))
    metrics_path = ROOT / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    best_name = metrics["best_model"] if metrics else "Not trained"
    best = metrics["models"][best_name] if metrics else None

    st.subheader("Project at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects", inventory["subject_count"])
    c2.metric("EEG recordings", inventory["file_count"])
    c3.metric("30-second epochs", f"{inventory['epoch_count']:,}")
    c4.metric("Held-out accuracy", f"{best['accuracy']:.1%}" if best else "—", help="Calculated on subjects kept out of training.")

    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.subheader("What the model does")
        st.write("Each 30-second EEG window is converted into 17 signal features, including Hjorth statistics and frequency-band power. Four classifiers are compared, and the selected model assigns one of five sleep-stage labels to each epoch.")
        st.markdown("**Stages:** Wake · N1 · N2 · N3 · REM")
        st.markdown("**Signal:** EEG Fpz-Cz · 100 Hz · 3,000 samples per epoch")
        if best:
            st.success(f"Selected classifier: **{best_name}** · grouped cross-validation macro F1 **{best['cross_validation_f1_macro_mean']:.3f}** · held-out macro F1 **{best['f1_macro']:.3f}**")
        else:
            st.warning("Model artifacts are not present. Run `python -m src.training` from the project root.")
    with right:
        st.subheader("Sleep-stage mix")
        if metrics:
            counts = pd.Series(metrics["class_counts"]).reindex(STAGE_ORDER)
            chart = px.pie(values=counts.values, names=counts.index, hole=.62,
                           color_discrete_sequence=["#194e5a", "#56a7a0", "#80c8b1", "#b6dfc5", "#e7c778"])
            chart.update_traces(textposition="inside", textinfo="percent", marker_line_color="#f5f8f7", marker_line_width=3)
            chart.update_layout(showlegend=True, legend_title_text="", margin=dict(l=0,r=0,t=10,b=0),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#172b3a"), height=300)
            st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
        else:
            st.caption("Train the models to see the stage distribution and evaluation summary.")

    st.subheader("Explore the project")
    steps = st.columns(4, gap="medium")
    cards = [
        ("01 · EXPLORE", "Dataset Analysis", "Inspect EEG waveforms, stage balance, and extracted features."),
        ("02 · PREDICT", "Sleep-Stage Prediction", "Upload a compatible preprocessed NPZ recording for epoch labels."),
        ("03 · COMPARE", "Model Comparison", "Compare four classifiers on held-out subjects and grouped CV."),
        ("04 · EVALUATE", "Model Evaluation", "Review confusion matrices, per-stage scores, and feature importance."),
    ]
    for col, (num, title, desc) in zip(steps, cards):
        with col:
            st.markdown(f'<div class="step-card"><div class="step-num">{num}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.caption("Evaluation uses subject-disjoint splits so epochs from the same person do not appear in both training and test sets. Results reflect a small 20-subject dataset and should not be interpreted as clinical performance.")
except (OSError, ValueError, KeyError, StopIteration) as exc:
    st.error(f"Dataset summary could not be loaded: {exc}")
