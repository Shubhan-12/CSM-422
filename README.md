# Sleep Cycle Prediction Using Machine Learning

This project classifies the five sleep stages **Wake, N1, N2, N3, and REM** from the preprocessed single-channel EEG recordings in the local `archive (11)/` folder. It does not use the former lifestyle CSV for training, prediction, or dashboard data. `Sleep_health_and_lifestyle_dataset.csv` is retained in the project as an inactive backup; no source dataset files are modified or downloaded by the application.

## Dataset inspection

The local archive contains **39 `.npz` recordings**, about 508 MB, from **20 subjects** (`SC400`–`SC419`). Subject `SC413` has one recording; the other subjects have two. Every recording provides:

- `x`: float32 EEG epochs shaped `(epochs, 3000, 1)` (the loader also accepts the squeezed 2-D view)
- `y`: one integer target for each epoch
- `fs`: 100 Hz sampling rate
- `ch_label`: `EEG Fpz-Cz`
- Header metadata arrays

Each window is 30 seconds. The observed label codes and measured distribution are:

| Code | Stage | Epochs |
|---:|---|---:|
| 0 | Wake | 8,285 |
| 1 | N1 | 2,804 |
| 2 | N2 | 17,799 |
| 3 | N3 | 5,703 |
| 4 | REM | 7,717 |
| | **Total** | **42,308** |

Code 3 is the consolidated N3 class (original sleep stages 3 and 4 are combined); code 4 is REM. The mapping agrees with the standard Sleep-EDF-20 class counts reported for single-channel Fpz-Cz recordings. The archive has no codes outside 0–4, so there are no movement or unknown epochs to remove. Inspection found no missing/non-finite EEG values and no exact duplicate EEG epochs. The label distribution is imbalanced, especially for N1.

Dataset source: [downloaded Kaggle archive](https://www.kaggle.com/datasets/fatemehmehrpuya/preprocessed-sleep-edf-20); Sleep-EDF source documentation: [PhysioNet Sleep-EDF Expanded](https://www.physionet.org/content/sleep-edfx/1.0.0/). The measured epoch totals match the published single-channel Sleep-EDF-20 counts ([reference dataset summary](https://personal.ntu.edu.sg/xlli/publication/sleep.pdf)).

## Signal features and model evaluation

`src/signal_processing.py` computes 17 deterministic features for each epoch: mean, standard deviation, RMS, Hjorth mobility and complexity, spectral entropy, dominant frequency, and absolute/relative delta, theta, alpha, sigma, and beta power. The same feature function processes training epochs and uploaded prediction files. Imputation and scaling are fitted within Scikit-learn pipelines on training folds only.

The subject ID is derived from each `SCxxx` recording prefix and is used only as a group key. A fixed stratified group holdout keeps all epochs from four subjects (`SC405`, `SC407`, `SC409`, `SC410`) in the test set and the other 16 subjects in training. No subject overlaps. Hyperparameter tuning uses four-fold `StratifiedGroupKFold` on training subjects only. Model choice uses mean training-CV macro F1; the test split is reserved for final reporting.

Four models are tuned and evaluated: Logistic Regression, Decision Tree, Random Forest, and a linear SVM (`LinearSVC`) to keep grouped tuning tractable for 42,308 epochs. Final measured results from the saved run:

| Model | Test accuracy | Test macro precision | Test macro recall | Test macro F1 | Train CV macro F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7291 | 0.6406 | 0.6768 | 0.6525 | 0.6356 ± 0.0485 |
| Decision Tree | 0.7260 | 0.6667 | 0.6963 | 0.6703 | 0.6336 ± 0.0297 |
| Random Forest | 0.7853 | 0.7043 | 0.7319 | 0.7152 | **0.6858 ± 0.0314** |
| SVM | 0.7585 | 0.6424 | 0.6722 | 0.6523 | 0.6389 ± 0.0295 |

Random Forest is selected by training-only cross-validation. Test scores are reported separately and do not select the model. The four-subject test set is small, and results may vary with a different subject split. The model predicts each epoch independently and does not use temporal context between neighboring epochs.

## Installation and run

From the project root, activate the existing virtual environment:

```bash
source .venv/bin/activate
```

To install or refresh dependencies:

```bash
python -m pip install -r requirements.txt
```

Train all models and regenerate metrics/artifacts:

```bash
python -m src.training
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

Open `notebooks/Sleep_Cycle_Prediction_ML.ipynb` in VS Code, select the **Python (sleep cycle .venv)** kernel, and run all cells for dataset inspection, signal plots, feature EDA, grouped training, evaluation, and a saved-model prediction demonstration. The committed workspace notebook has been executed with outputs from `archive (11)/`. The former `Sleep_Disorder_Model_Evaluation.ipynb` contains only a pointer to the current notebook so its old results are not mistaken for EEG results.

## Prediction page input

Upload a preprocessed `.npz` file with `x`, `fs`, and `ch_label` arrays. It must contain 30-second epochs of 3,000 samples at 100 Hz. The app displays one stage prediction per epoch and the selected Random Forest's probability estimates for the first epoch. Unsupported sampling rates, epoch lengths, and file formats receive a clear error. Predictions are experimental and are not medical diagnoses.

## Project structure

```text
app.py
pages/
  1_💤_Sleep_Stage_Prediction.py
  2_📊_Data_Analysis.py
  3_⚖️_Model_Comparison.py
  4_🧪_Model_Evaluation.py
notebooks/
  Sleep_Cycle_Prediction_ML.ipynb
src/
  data_loading.py       # Local NPZ inventory, schema validation, and labels
  preprocessing.py      # Stage mapping and feature validation
  signal_processing.py  # Shared EEG feature extraction
  training.py           # Subject-grouped tuning, evaluation, and saving
  evaluation.py         # Evaluation helpers
  prediction.py         # Saved pipeline and NPZ prediction
models/
  sleep_stage_model.joblib
  logistic_regression.joblib
  decision_tree.joblib
  random_forest.joblib
  svm.joblib
  metrics.json
  model_comparison.csv
archive (11)/          # Existing local dataset; unchanged
```

## Limitations

This project evaluates only the provided 20-subject Sleep-EDF subset and one EEG channel. The test set covers four subjects, so it cannot establish broad clinical performance. The exported preprocessed NPZ files have already been epoch-segmented and labeled; the app expects this same representation rather than arbitrary raw EDF files. The models are educational experiments and must not be used for diagnosis or treatment decisions.
