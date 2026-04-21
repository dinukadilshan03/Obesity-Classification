# Obesity-Classification

A production-style machine learning project that predicts obesity class from lifestyle and physical attributes, with **96.22% test accuracy** using a tuned Logistic Regression pipeline.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://obesity-classification-7r3s7gmmohdf3yhobxhysy.streamlit.app/)

---

## 1) Executive Summary

This project builds and deploys an end-to-end ML pipeline for multiclass obesity classification (7 classes), from EDA to preprocessing, model tuning, evaluation, and Streamlit deployment.

**Key results (best notebook run):**
- Test Accuracy: **96.22%**
- CV Accuracy (GridSearchCV best): **95.50%**
- Macro Precision / Recall / F1: **0.96 / 0.96 / 0.96**

---

## 2) Problem Statement

Obesity risk is influenced by multiple factors (physical metrics, diet, and behavior). A reliable classifier helps quickly stratify individuals into risk categories for early intervention, triage, and awareness workflows.

**Business case / value:**
- Faster risk screening support
- Consistent, data-driven categorization
- Deployable UI for non-technical users

---

## 3) Solution Overview

High-level approach:
1. Load obesity dataset
2. Perform EDA and stratified split
3. Engineer BMI feature (`Weight / Height²`)
4. Apply mixed preprocessing (scaling + encoding)
5. Train/tune Logistic Regression with GridSearchCV
6. Evaluate with classification report + confusion matrix
7. Deploy trained pipeline in Streamlit app

---

## 4) Dataset Description

- **File in repo:** `data/ObesityDataSet_raw.csv`
- **Rows:** 2,111
- **Columns:** 17 total (16 predictors + 1 target)
- **Target column:** `NObeyesdad`
- **Classes:**
  - Insufficient_Weight (272)
  - Normal_Weight (287)
  - Overweight_Level_I (290)
  - Overweight_Level_II (290)
  - Obesity_Type_I (351)
  - Obesity_Type_II (297)
  - Obesity_Type_III (324)

**Feature groups:**
- Physical: `Gender`, `Age`, `Height`, `Weight`
- Dietary: `FAVC`, `FCVC`, `NCP`, `CAEC`, `CH2O`, `CALC`
- Lifestyle/History: `family_history_with_overweight`, `SMOKE`, `SCC`, `FAF`, `TUE`, `MTRANS`

---

## 5) Feature Engineering

- Custom transformer in `src/preprocessing.py` adds:
  - **BMI = Weight / (Height²)**
- BMI is included in numeric preprocessing and model training pipeline.

---

## 6) Exploratory Data Analysis (EDA)

Notebook: `notebooks/EDA.ipynb`

Key outputs include:
- Stratified train/test split to avoid leakage
- Class distribution checks
- Feature distribution and relationship inspection
- Inputs for downstream preprocessing design

---

## 7) Methodology

- **Task:** Multiclass classification (7 obesity classes)
- **Primary algorithm:** Logistic Regression
- **Why Logistic Regression:**
  - Strong baseline for structured tabular data
  - Fast training/inference
  - Interpretable coefficients (relative feature influence)
  - High observed accuracy after tuning (96.22%)

---

## 8) Model Performance

From `notebooks/Evaluate.ipynb` (optimized model run):

| Metric | Value |
|---|---:|
| Accuracy | **0.9622** |
| Macro Precision | 0.96 |
| Macro Recall | 0.96 |
| Macro F1-score | 0.96 |
| Weighted Precision | 0.96 |
| Weighted Recall | 0.96 |
| Weighted F1-score | 0.96 |

**Confusion Matrix:** Generated in evaluation notebook via `ConfusionMatrixDisplay.from_predictions(...)`.

---

## 9) Hyperparameter Tuning

Notebook: Tuning workflow in `notebooks/Tuninig.ipynb`

**Tool:** `GridSearchCV(cv=5, scoring='accuracy')`

**Parameter grid:**
- `classifier__C`: `[0.1, 1.0, 10.0, 100.0]`
- `classifier__max_iter`: `[1000, 2000]`
- `classifier__solver`: `['lbfgs', 'saga']`

**Best result:**
- Best CV score: **0.9550**
- Best params: `{'classifier__C': 100.0, 'classifier__max_iter': 1000, 'classifier__solver': 'lbfgs'}`

---

## 10) Data Preprocessing Pipeline

Implemented with `Pipeline` + `ColumnTransformer`:

1. **Feature engineering**: BMI via `FunctionTransformer`
2. **Numeric features** (`Age`, `Height`, `Weight`, `BMI`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`)
   - `RobustScaler`
3. **Ordinal features** (`CAEC`, `CALC`)
   - `OrdinalEncoder` with order: `['no', 'Sometimes', 'Frequently', 'Always']`
4. **Nominal feature** (`MTRANS`)
   - `OneHotEncoder(handle_unknown='ignore')`
5. **Binary categoricals** (`Gender`, `FAVC`, `SMOKE`, `SCC`, `family_history_with_overweight`)
   - Encoded in categorical branch (from notebook pipeline)

---

## 11) Project Structure

```text
Obesity-Classification/
├── assets/
│   └── app-screenshot.png               # Streamlit UI screenshot
├── data/
│   └── ObesityDataSet_raw.csv           # Raw dataset
├── models/
│   ├── obesity_classifier_pipeline.pkl
│   └── obesity_classifier_v2_optimized.pkl
├── notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   ├── Tuninig.ipynb
│   └── Evaluate.ipynb
├── src/
│   ├── app.py                           # Streamlit app
│   └── preprocessing.py                 # BMI transformer
├── main.py                              # Basic entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 12) Setup & Installation

```bash
# clone
git clone https://github.com/dinukadilshan03/Obesity-Classification.git
cd Obesity-Classification

# create venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# install dependencies
pip install -r requirements.txt
```

Optional (uv users):
```bash
uv sync
```

---

## 13) Running the Model

### A) Reproduce training/evaluation flow
Run notebooks in order:
1. `notebooks/EDA.ipynb`
2. `notebooks/Preprocessing.ipynb`
3. Tuning notebook: `notebooks/Tuninig.ipynb`
4. `notebooks/Evaluate.ipynb`

### B) Use saved model artifact
The deployed app loads `models/obesity_classifier_v2_optimized.pkl`.

---

## 14) Web Application

- **Framework:** Streamlit
- **Entry point:** `src/app.py`
- **Run locally:**

```bash
streamlit run src/app.py
```

**Live app links:**
- https://obesity-classification-7r3s7gmmohdf3yhobxhysy.streamlit.app/
- https://obesity-classification-pipeline.streamlit.app

---

## 15) Prediction Examples

Sample input schema (from app form):

```json
{
  "Gender": "Male",
  "Age": 25,
  "Height": 1.75,
  "Weight": 75.0,
  "family_history_with_overweight": "yes",
  "FAVC": "yes",
  "FCVC": 2.0,
  "NCP": 3.0,
  "CAEC": "Sometimes",
  "SMOKE": "no",
  "CH2O": 2.0,
  "SCC": "no",
  "FAF": 1.0,
  "TUE": 1.0,
  "CALC": "Sometimes",
  "MTRANS": "Walking"
}
```

Example output format:
- `Predicted Class: Normal_Weight` (or one of the 7 class labels)

---

## 16) Model Artifacts

Stored in `models/`:
- `obesity_classifier_v2_optimized.pkl` (optimized deployed model)
- `obesity_classifier_pipeline.pkl` (pipeline artifact)

Artifacts are serialized with `joblib` and loaded at inference time in `src/app.py`.

---

## 17) Notebooks

- `EDA.ipynb`: class/feature exploration and split strategy
- `Preprocessing.ipynb`: transformer/pipeline construction
- Tuning notebook (`Tuninig.ipynb`): GridSearchCV tuning and model selection
- `Evaluate.ipynb`: classification report and confusion matrix

---

## 18) Performance Metrics (ROC, Importance, Tables)

- **Metrics table:** Included in this README and detailed in `Evaluate.ipynb`
- **Confusion matrix:** Available in `Evaluate.ipynb`
- **ROC curve (multiclass):** Recommended as One-vs-Rest extension in evaluation workflow
- **Feature importance:** For Logistic Regression, coefficient magnitude can be used as a feature-impact proxy

> Current repository artifacts prioritize classification report + confusion matrix; ROC/feature-impact plots can be added in `Evaluate.ipynb` as a direct extension.

---

## 19) Optimization Strategies

- Stratified split to preserve class ratios
- 5-fold cross-validation (`GridSearchCV`)
- Search over regularization strength, solver, and iteration limits
- End-to-end sklearn pipeline to reduce leakage and ensure reproducibility

---

## 20) Deployment

### Live deployment
- Streamlit Cloud app available at the links above.

### Deploy your own
1. Push project to GitHub
2. Create a Streamlit Cloud app pointing to `src/app.py`
3. Ensure `requirements.txt` is present
4. Add model files under `models/`
5. Deploy and validate prediction flow

---

## 21) Future Improvements

- Add ensemble benchmarks (RandomForest/XGBoost/stacking)
- Add probability calibration and uncertainty reporting
- Add richer monitoring/dashboard metrics
- Add robust automated tests for preprocessing + inference paths
- Add ROC/PR and coefficient-importance visuals to evaluation artifacts

---

## 22) Learning Outcomes

This project demonstrates:
- End-to-end ML workflow execution
- Feature engineering for tabular healthcare-style data
- Reproducible sklearn pipeline design
- Hyperparameter tuning with cross-validation
- Model evaluation and practical deployment via Streamlit

---

## 23) License & Author

- **License:** MIT (as stated in repository documentation)
- **Author:** [@dinukadilshan03](https://github.com/dinukadilshan03)

---

## Screenshot

![Obesity Classification App](assets/app-screenshot.png)
