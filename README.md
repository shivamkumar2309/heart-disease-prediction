# Heart Disease Prediction & Analysis

> **End-to-end machine learning pipeline** for predicting cardiovascular disease risk using the UCI Heart Disease dataset.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Features](#features)
5. [How to Run](#how-to-run)
6. [Machine Learning Models](#machine-learning-models)
7. [Visualizations](#visualizations)
8. [Patient Risk Prediction API](#patient-risk-prediction-api)
9. [Results Summary](#results-summary)
10. [Feature Descriptions](#feature-descriptions)

---

## Project Overview

This project provides a complete, production-ready implementation for **detecting and analyzing heart disease risk** in patients. It covers:

- Exploratory data analysis (EDA) with demographic and clinical breakdowns
- Feature correlation and risk factor identification
- Three machine learning classifiers with rigorous evaluation
- 10 publication-quality visualizations saved automatically
- A callable `predict_patient_risk()` function for real-time inference

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — UCI Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **File name** | `heart.csv` |
| **Rows** | ~1,025 patient records |
| **Features** | 13 clinical attributes + 1 target |
| **Target** | `1` = Heart Disease present · `0` = Absent |

### How to obtain the data

1. Go to <https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset>
2. Click **Download** and extract the archive.
3. Place `heart.csv` in the **project root directory** (same folder as `heart_disease_prediction.py`).

---

## Project Structure

```
heart-disease-prediction/
├── heart_disease_prediction.py   # Main pipeline (EDA + ML + predictions)
├── heart.csv                     # Dataset (download separately)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── outputs/                      # Auto-created: all saved plots
    ├── 01_disease_distribution.png
    ├── 02_age_distribution.png
    ├── 03_gender_stats.png
    ├── 04_correlation_heatmap.png
    ├── 05_health_metrics.png
    ├── 06_feature_importance.png
    ├── 07_confusion_matrices.png
    ├── 08_roc_auc_curves.png
    ├── 09_risk_segmentation.png
    └── 10_model_comparison.png
```

---

## Features

### Data Analysis
- **Age distribution** — disease prevalence across 10-year age bins
- **Gender breakdown** — male vs. female disease statistics
- **Health metric comparison** — BP, cholesterol, and max heart rate vs. disease status
- **Correlation heatmap** — pairwise feature relationships
- **Risk factor ranking** — features ranked by correlation strength with target

### Machine Learning
- 80/20 stratified train/test split
- StandardScaler normalization inside sklearn Pipelines
- 5-fold stratified cross-validation
- Three classifiers: Logistic Regression, Random Forest, SVM
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Predictions
- `predict_patient_risk()` function accepting 13 clinical inputs
- Returns a **risk score (0–100%)**, risk level, and human-readable interpretation

---

## How to Run

### 1. Clone / download this project
```bash
git clone <repo-url>
cd heart-disease-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `heart.csv` in the project root (see [Dataset](#dataset) section above).

### 5. Run the pipeline
```bash
python heart_disease_prediction.py
```

Or specify a custom path:
```bash
python heart_disease_prediction.py path/to/heart.csv
```

All output plots are saved to the `outputs/` directory automatically.

---

## Machine Learning Models

| Model | Description |
|---|---|
| **Logistic Regression** | Linear baseline; interpretable coefficients; L2 regularisation (C=1.0) |
| **Random Forest** | Ensemble of 200 decision trees; provides feature importances; handles non-linearity |
| **SVM** | RBF kernel; effective in high-dimensional spaces; probability calibration enabled |

All three models are trained within `sklearn.pipeline.Pipeline` objects that include `StandardScaler` to prevent data leakage.

---

## Visualizations

| File | Description |
|---|---|
| `01_disease_distribution.png` | Pie chart — overall disease vs. no-disease split |
| `02_age_distribution.png` | Grouped bar — disease count by age bracket |
| `03_gender_stats.png` | Grouped bar — disease count by gender |
| `04_correlation_heatmap.png` | Lower-triangle heatmap of all feature correlations |
| `05_health_metrics.png` | Box plots — BP, cholesterol, max HR vs. disease status |
| `06_feature_importance.png` | Horizontal bar — Random Forest feature importances |
| `07_confusion_matrices.png` | Side-by-side confusion matrices for all 3 models |
| `08_roc_auc_curves.png` | Multi-model ROC curves on one plot |
| `09_risk_segmentation.png` | Scatter — patient age vs. predicted risk probability (3 tiers) |
| `10_model_comparison.png` | Grouped bar — Accuracy, Precision, Recall, F1, AUC per model |

---

## Patient Risk Prediction API

Use `predict_patient_risk()` to score a new patient in real time:

```python
from heart_disease_prediction import predict_patient_risk, load_data, preprocess_data, train_and_evaluate
from sklearn.model_selection import train_test_split

# Build the model
df = load_data("heart.csv")
X, y, _ = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
results, rf_pipeline, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, X.columns.tolist())

# Predict for a new patient
result = predict_patient_risk(
    model     = rf_pipeline,
    age       = 55,
    sex       = 1,       # 1 = Male
    cp        = 2,       # Chest pain type
    trestbps  = 140,     # Resting BP (mm Hg)
    chol      = 250,     # Cholesterol (mg/dl)
    fbs       = 0,       # Fasting blood sugar > 120? No
    restecg   = 1,       # Resting ECG result
    thalach   = 150,     # Max heart rate
    exang     = 1,       # Exercise-induced angina? Yes
    oldpeak   = 1.5,     # ST depression
    slope     = 1,       # Slope of peak ST segment
    ca        = 1,       # Major vessels coloured
    thal      = 2,       # Thalassemia type
)

print(result["interpretation"])
# Risk Score: 74.0% — High Risk. The model indicates a high probability of heart disease ...
```

### Return value structure

```python
{
    "risk_score":     74.0,           # float, 0–100 (%)
    "risk_level":     "High",         # "Low" | "Moderate" | "High"
    "interpretation": "Risk Score: 74.0% — High Risk. ...",
    "prediction":     1,              # 1 = Disease, 0 = No Disease
}
```

### Risk tier thresholds

| Risk Level | Score Range | Recommendation |
|---|---|---|
| 🟢 Low | 0 – 33% | Healthy lifestyle; routine check-ups |
| 🟡 Moderate | 33 – 66% | Physician consultation advised |
| 🔴 High | 66 – 100% | Immediate medical attention recommended |

---

## Results Summary

> Actual metrics vary slightly with dataset version. Values below are representative.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.85 | ~0.86 | ~0.87 | ~0.86 | ~0.92 |
| **Random Forest** | **~0.99** | **~0.99** | **~0.99** | **~0.99** | **~1.00** |
| SVM | ~0.87 | ~0.88 | ~0.89 | ~0.88 | ~0.94 |

**Key findings:**
- Random Forest achieves near-perfect classification on this dataset.
- The most predictive features are: `cp` (chest pain type), `thal`, `ca`, `oldpeak`, `thalach`.
- Heart disease is slightly more prevalent in male patients and in the 46–65 age range.
- Patients with exercise-induced angina (`exang = 1`) show significantly higher disease rates.
- Maximum heart rate (`thalach`) is inversely correlated with disease — lower HR associates with higher risk.

---

## Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `age` | int | Age in years |
| `sex` | binary | 1 = Male · 0 = Female |
| `cp` | categorical (0–3) | Chest pain type (0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic) |
| `trestbps` | int | Resting blood pressure (mm Hg) |
| `chol` | int | Serum cholesterol (mg/dl) |
| `fbs` | binary | Fasting blood sugar > 120 mg/dl (1/0) |
| `restecg` | categorical (0–2) | Resting ECG (0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy) |
| `thalach` | int | Maximum heart rate achieved (bpm) |
| `exang` | binary | Exercise-induced angina (1 = Yes · 0 = No) |
| `oldpeak` | float | ST depression induced by exercise relative to rest |
| `slope` | categorical (0–2) | Slope of peak exercise ST segment |
| `ca` | int (0–4) | Number of major vessels coloured by fluoroscopy |
| `thal` | categorical | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversable defect) |
| `target` | binary | **Label** — 1 = Disease · 0 = No Disease |

---

## License

This project is provided for educational and research purposes.  
Dataset credits: [Kaggle — UCI Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (originally from UCI ML Repository).