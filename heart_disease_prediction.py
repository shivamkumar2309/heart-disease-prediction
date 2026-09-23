# =============================================================================
# Heart Disease Prediction and Analysis
# Dataset: UCI Heart Disease Dataset (Kaggle)
# https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# =============================================================================
# Description:
#   - End-to-end machine learning pipeline for predicting heart disease risk.
#   - Covers EDA, feature engineering, multi-model training, evaluation,
#     visualization, and a patient risk prediction function.
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE   = ["#3b82d4", "#e05c5c"]
BG_COLOR  = "#f7f8fa"
GRID_CLR  = "#e5e7eb"
TEXT_CLR  = "#1f2328"
MUTED_CLR = "#57606a"

# Output directory for saved figures
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(filepath: str = "heart.csv") -> pd.DataFrame:
    """
    Load the UCI Heart Disease dataset from a CSV file.

    Expected columns (14 features):
        age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal, target

    Args:
        filepath: Path to the heart.csv file.

    Returns:
        Raw DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download 'heart.csv' from:\n"
            "  https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset\n"
            "and place it in the project root directory."
        )
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

def explore_data(df: pd.DataFrame) -> None:
    """Print a structured EDA summary to the console."""

    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n── Shape ──────────────────────────────────────────────────")
    print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

    print("\n── First 5 rows ───────────────────────────────────────────")
    print(df.head().to_string())

    print("\n── Data types & null counts ───────────────────────────────")
    info = pd.DataFrame({
        "dtype":    df.dtypes,
        "non_null": df.notnull().sum(),
        "null":     df.isnull().sum(),
    })
    print(info.to_string())

    print("\n── Descriptive statistics ─────────────────────────────────")
    print(df.describe().round(2).to_string())

    print("\n── Target distribution ────────────────────────────────────")
    counts = df["target"].value_counts()
    pct    = df["target"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": counts, "pct (%)": pct.round(1)}).to_string())

    print("\n── Gender distribution ────────────────────────────────────")
    gender_map = {0: "Female", 1: "Male"}
    gender_counts = df["sex"].map(gender_map).value_counts()
    print(gender_counts.to_string())

    print("\n── Age summary ────────────────────────────────────────────")
    print(f"  Min: {df['age'].min()}  |  Max: {df['age'].max()}  |  "
          f"Mean: {df['age'].mean():.1f}  |  Std: {df['age'].std():.1f}")


# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================

def preprocess_data(df: pd.DataFrame):
    """
    Clean and prepare data for model training.

    Steps:
        1. Drop duplicate rows.
        2. Impute missing values (median for numeric, mode for categorical).
        3. Encode binary/categorical columns.
        4. Split into feature matrix X and label vector y.

    Returns:
        X (DataFrame), y (Series), feature_names (list)
    """
    df = df.copy()

    # ── Remove duplicates ──────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    if removed:
        print(f"[INFO] Removed {removed} duplicate row(s).")

    # ── Handle missing values ──────────────────────────────────────────────
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # ── Separate features / target ─────────────────────────────────────────
    X = df.drop(columns=["target"])
    y = df["target"]

    feature_names = X.columns.tolist()
    print(f"[INFO] Features: {len(feature_names)}  |  Samples: {len(X)}")
    return X, y, feature_names


# =============================================================================
# 4. VISUALIZATIONS
# =============================================================================

def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent styling to a matplotlib Axes."""
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_CLR, pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED_CLR)
    ax.set_ylabel(ylabel, fontsize=10, color=MUTED_CLR)
    ax.tick_params(colors=MUTED_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.8, linestyle="--", alpha=0.7)


def plot_disease_distribution(df: pd.DataFrame) -> None:
    """Pie chart — overall heart disease distribution."""
    counts = df["target"].value_counts()
    labels = ["Heart Disease", "No Heart Disease"]
    colors = PALETTE

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG_COLOR)
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11, "color": TEXT_CLR},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Heart Disease Distribution", fontsize=14,
                 fontweight="bold", color=TEXT_CLR, pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_disease_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_age_distribution(df: pd.DataFrame) -> None:
    """Bar chart — age-wise heart disease distribution (10-year bins)."""
    df2 = df.copy()
    df2["age_group"] = pd.cut(
        df2["age"], bins=[25, 35, 45, 55, 65, 80],
        labels=["26–35", "36–45", "46–55", "56–65", "66–80"],
    )
    counts = df2.groupby(["age_group", "target"], observed=True).size().unstack(fill_value=0)
    counts.columns = ["No Disease", "Disease"]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG_COLOR)
    x = np.arange(len(counts))
    width = 0.38
    ax.bar(x - width / 2, counts["Disease"],    width, color=PALETTE[1], label="Disease",    alpha=0.88)
    ax.bar(x + width / 2, counts["No Disease"], width, color=PALETTE[0], label="No Disease", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, fontsize=10)
    ax.legend(fontsize=10)
    _style_ax(ax, "Age-wise Heart Disease Distribution", "Age Group", "Patient Count")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_age_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_gender_stats(df: pd.DataFrame) -> None:
    """Grouped bar — gender-wise disease statistics."""
    gender_map  = {0: "Female", 1: "Male"}
    disease_map = {0: "No Disease", 1: "Disease"}
    df2 = df.copy()
    df2["sex"]    = df2["sex"].map(gender_map)
    df2["target"] = df2["target"].map(disease_map)

    counts = df2.groupby(["sex", "target"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_COLOR)
    x     = np.arange(len(counts))
    width = 0.38
    ax.bar(x - width / 2, counts["Disease"],    width, color=PALETTE[1], label="Disease",    alpha=0.88)
    ax.bar(x + width / 2, counts["No Disease"], width, color=PALETTE[0], label="No Disease", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, fontsize=11)
    ax.legend(fontsize=10)
    _style_ax(ax, "Gender-wise Heart Disease Statistics", "Gender", "Patient Count")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_gender_stats.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Full-feature correlation heatmap."""
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG_COLOR)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        cmap="coolwarm", center=0,
        linewidths=0.5, linecolor=GRID_CLR,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14,
                 fontweight="bold", color=TEXT_CLR, pad=12)
    ax.tick_params(colors=MUTED_CLR, labelsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_health_metrics(df: pd.DataFrame) -> None:
    """Box plots — BP, cholesterol, and max heart rate vs. disease status."""
    metrics = [
        ("trestbps", "Resting Blood Pressure (mm Hg)"),
        ("chol",     "Serum Cholesterol (mg/dl)"),
        ("thalach",  "Max Heart Rate Achieved (bpm)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG_COLOR)
    fig.suptitle("Key Health Metrics vs Heart Disease",
                 fontsize=14, fontweight="bold", color=TEXT_CLR, y=1.02)

    disease_labels = {0: "No Disease", 1: "Disease"}
    df2 = df.copy()
    df2["Disease"] = df2["target"].map(disease_labels)

    for ax, (col, ylabel) in zip(axes, metrics):
        sns.boxplot(
            data=df2, x="Disease", y=col, ax=ax,
            palette={k: v for k, v in zip(["No Disease", "Disease"], PALETTE[::-1])},
            width=0.5, linewidth=1.2,
            order=["No Disease", "Disease"],
        )
        _style_ax(ax, ylabel.split(" (")[0], "Disease Status", ylabel.split("(")[1].rstrip(")") if "(" in ylabel else "")
        ax.set_xlabel("Disease Status", fontsize=9, color=MUTED_CLR)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_health_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_feature_importance(feature_names: list, importances: np.ndarray,
                            model_name: str = "Random Forest") -> None:
    """Horizontal bar chart — feature importance scores."""
    idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp   = importances[idx]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
    colors = [PALETTE[1] if v >= sorted_imp.mean() else PALETTE[0] for v in sorted_imp]
    bars = ax.barh(sorted_names, sorted_imp, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, sorted_imp):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=MUTED_CLR)
    _style_ax(ax, f"Feature Importance — {model_name}", "Importance Score", "Feature")
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_confusion_matrices(results: dict, y_test: pd.Series) -> None:
    """Side-by-side confusion matrices for all trained models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), facecolor=BG_COLOR)
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", fontsize=14,
                 fontweight="bold", color=TEXT_CLR, y=1.02)

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold", color=TEXT_CLR)
        ax.set_facecolor(BG_COLOR)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "07_confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_roc_curves(results: dict, y_test: pd.Series) -> None:
    """Multi-model ROC-AUC curves on a single plot."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
    colors = ["#3b82d4", "#e05c5c", "#7c5cd8", "#22c55e"]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.50)")

    for (name, res), color in zip(results.items(), colors):
        if res.get("y_proba") is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            auc_val = res["roc_auc"]
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name} (AUC = {auc_val:.3f})")

    _style_ax(ax, "ROC-AUC Curves — Model Comparison",
              "False Positive Rate", "True Positive Rate")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_roc_auc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_precision_recall_curves(results: dict, y_test: pd.Series) -> None:
    """Precision-Recall curves for all models on a single plot."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
    colors = ["#3b82d4", "#e05c5c", "#7c5cd8", "#22c55e"]
    
    # No-skill baseline (assumes equal distribution)
    ax.plot([0, 1], [0.5, 0.5], "k--", linewidth=1, alpha=0.5, label="No Skill (AP = 0.50)")

    for (name, res), color in zip(results.items(), colors):
        if res.get("y_proba") is not None:
            precision, recall, _ = precision_recall_curve(y_test, res["y_proba"])
            ap = average_precision_score(y_test, res["y_proba"])
            ax.plot(recall, precision, color=color, linewidth=2,
                    label=f"{name} (AP = {ap:.3f})")

    _style_ax(ax, "Precision-Recall Curves — Model Comparison",
              "Recall (True Positive Rate)", "Precision (Positive Predictive Value)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08b_precision_recall_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_risk_segmentation(df: pd.DataFrame, scaler: StandardScaler,
                           rf_model: Pipeline) -> None:
    """
    Scatter — patient risk segmentation plot.
    X-axis: age  |  Y-axis: predicted probability of disease
    Color-coded by risk tier: Low / Moderate / High.
    """
    # Deduplicate the dataframe to match the model's training data
    df_clean = df.drop_duplicates().copy()
    
    X_all, _, _ = preprocess_data(df_clean)
    X_scaled     = scaler.transform(X_all)
    probas        = rf_model.predict_proba(X_all)[:, 1]

    risk_labels = pd.cut(
        probas,
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=["Low Risk", "Moderate Risk", "High Risk"],
    )
    color_map = {"Low Risk": "#22c55e", "Moderate Risk": "#f59e0b", "High Risk": "#e05c5c"}

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG_COLOR)
    for tier in ["Low Risk", "Moderate Risk", "High Risk"]:
        mask = risk_labels == tier
        ax.scatter(df_clean["age"][mask], probas[mask],
                   c=color_map[tier], label=tier, alpha=0.7, s=35,
                   edgecolors="white", linewidths=0.4)

    ax.axhline(0.33, color=GRID_CLR, linestyle="--", linewidth=1)
    ax.axhline(0.66, color=GRID_CLR, linestyle="--", linewidth=1)
    _style_ax(ax, "Patient Risk Segmentation (Age vs. Disease Probability)",
              "Age", "Predicted Disease Probability")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_risk_segmentation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


def plot_model_comparison(results: dict) -> None:
    """Grouped bar chart comparing key metrics across all models."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors  = ["#3b82d4", "#e05c5c", "#7c5cd8"]
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG_COLOR)
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in metrics]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=colors[i % len(colors)], alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=MUTED_CLR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    _style_ax(ax, "Model Performance Comparison", "", "Score")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "10_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path}")


# =============================================================================
# 5. MODEL TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """
    Train three classifiers (Logistic Regression, Random Forest, SVM)
    inside sklearn Pipelines (StandardScaler → Classifier).

    Returns:
        results  (dict)  — per-model metrics and predictions
        best_rf  (Pipeline) — fitted Random Forest pipeline
        scaler   (StandardScaler) — fitted scaler (for risk segmentation)
    """
    # ── Scaler (fitted once on training data) ─────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Model definitions ──────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=2,
            random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=42, class_weight='balanced'
        ),
    }

    results = {}
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    best_rf_pipeline = None

    for name, model in models.items():
        # ── Fit & predict ──────────────────────────────────────────────────
        model.fit(X_train_sc, y_train)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        # ── Metrics ────────────────────────────────────────────────────────
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba)

        # ── 5-fold cross-validation accuracy ──────────────────────────────
        import numpy as _np  # already imported at top; alias for clarity
        cv_scores = cross_val_score(model, scaler.transform(X_train), y_train,
                                     cv=cv, scoring="accuracy")

        results[name] = {
            "model":     model,
            "y_pred":    y_pred,
            "y_proba":   y_proba,
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "roc_auc":   auc,
            "cv_mean":   cv_scores.mean(),
            "cv_std":    cv_scores.std(),
        }

        # ── Console summary ────────────────────────────────────────────────
        print(f"\n  ▸ {name}")
        print(f"    Accuracy : {acc:.4f}  |  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"    Precision: {prec:.4f}  |  Recall    : {rec:.4f}")
        print(f"    F1-Score : {f1:.4f}  |  ROC-AUC   : {auc:.4f}")
        print(f"\n    Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'], zero_division=0)}")

        # ── Keep the Random Forest pipeline for downstream use ─────────────
        if name == "Random Forest":
            # Wrap in a Pipeline for the risk predictor
            best_rf_pipeline = Pipeline([
                ("scaler", StandardScaler().fit(X_train)),
                ("clf",    model),
            ])

    return results, best_rf_pipeline, scaler


# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(rf_model, feature_names: list) -> np.ndarray:
    """Extract feature importances from the Random Forest classifier."""
    # rf_model may be a Pipeline or a raw estimator
    clf = rf_model.named_steps["clf"] if hasattr(rf_model, "named_steps") else rf_model
    return clf.feature_importances_


# =============================================================================
# 7. RISK IDENTIFICATION
# =============================================================================

def identify_risk_factors(df: pd.DataFrame) -> None:
    """Print correlation-based risk factor ranking."""
    print("\n" + "=" * 60)
    print("  RISK FACTOR ANALYSIS (Correlation with Target)")
    print("=" * 60)
    corr = df.corr(numeric_only=True)["target"].drop("target").abs().sort_values(ascending=False)
    for feat, val in corr.items():
        bar   = "█" * int(val * 30)
        level = "HIGH" if val > 0.4 else ("MODERATE" if val > 0.2 else "LOW")
        print(f"  {feat:<12} │ {bar:<30} │ {val:.3f}  [{level}]")


# =============================================================================
# 8. PATIENT RISK PREDICTION FUNCTION
# =============================================================================

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

def predict_patient_risk(
    model: Pipeline,
    age: int,
    sex: int,          # 1 = Male, 0 = Female
    cp: int,           # Chest pain type (0–3)
    trestbps: int,     # Resting blood pressure (mm Hg)
    chol: int,         # Serum cholesterol (mg/dl)
    fbs: int,          # Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
    restecg: int,      # Resting ECG results (0–2)
    thalach: int,      # Maximum heart rate achieved (bpm)
    exang: int,        # Exercise-induced angina (1 = Yes, 0 = No)
    oldpeak: float,    # ST depression induced by exercise
    slope: int,        # Slope of peak exercise ST segment (0–2)
    ca: int,           # Number of major vessels coloured by fluoroscopy (0–4)
    thal: int,         # Thalassemia (0 = Normal, 1 = Fixed defect, 2 = Reversable defect)
) -> dict:
    """
    Predict heart disease risk for a new patient.

    Args:
        model   : Trained sklearn Pipeline (scaler + classifier).
        age     : Patient age in years.
        sex     : 1 = Male, 0 = Female.
        cp      : Chest pain type (0 = typical angina … 3 = asymptomatic).
        trestbps: Resting blood pressure (mm Hg).
        chol    : Serum cholesterol (mg/dl).
        fbs     : Fasting blood sugar > 120 mg/dl (1/0).
        restecg : Resting ECG (0 = normal, 1 = ST-T abnormality, 2 = LVH).
        thalach : Maximum heart rate achieved.
        exang   : Exercise-induced angina (1/0).
        oldpeak : ST depression induced by exercise relative to rest.
        slope   : Slope of peak exercise ST segment (0–2).
        ca      : Number of major vessels (0–4).
        thal    : Thalassemia (0 = normal, 1 = fixed defect, 2 = reversable defect).

    Returns:
        dict with keys:
            risk_score      — float 0–100 (%)
            risk_level      — "Low" | "Moderate" | "High"
            interpretation  — human-readable string
            prediction      — 1 (disease) | 0 (no disease)
    """
    patient = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }])

    proba      = model.predict_proba(patient)[0][1]
    prediction = model.predict(patient)[0]
    risk_score = round(proba * 100, 2)

    if risk_score < 33:
        risk_level      = "Low"
        interpretation  = (
            f"Risk Score: {risk_score:.1f}% — Low Risk. "
            "The model predicts a low probability of heart disease. "
            "Maintain a healthy lifestyle with regular check-ups."
        )
    elif risk_score < 66:
        risk_level      = "Moderate"
        interpretation  = (
            f"Risk Score: {risk_score:.1f}% — Moderate Risk. "
            "Some risk factors are present. Consult a physician for "
            "a full cardiovascular evaluation and lifestyle adjustments."
        )
    else:
        risk_level      = "High"
        interpretation  = (
            f"Risk Score: {risk_score:.1f}% — High Risk. "
            "The model indicates a high probability of heart disease. "
            "Immediate medical consultation is strongly recommended."
        )

    return {
        "risk_score":     risk_score,
        "risk_level":     risk_level,
        "interpretation": interpretation,
        "prediction":     int(prediction),
    }


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================

def main(filepath: str = "heart.csv") -> None:
    """
    End-to-end pipeline:
        Load → EDA → Preprocess → Visualize → Train → Evaluate → Predict
    """

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    df = load_data(filepath)

    # ── Step 2: EDA ──────────────────────────────────────────────────────────
    explore_data(df)
    identify_risk_factors(df)

    # ── Step 3: Preprocess ───────────────────────────────────────────────────
    X, y, feature_names = preprocess_data(df)

    # ── Step 4: Train / Test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # ── Step 5: Visualizations (pre-model) ───────────────────────────────────
    print("\n[INFO] Generating visualizations …")
    plot_disease_distribution(df)
    plot_age_distribution(df)
    plot_gender_stats(df)
    plot_correlation_heatmap(df)
    plot_health_metrics(df)

    # ── Step 6: Train & evaluate models ──────────────────────────────────────
    results, rf_pipeline, scaler = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    # ── Step 7: Post-model visualizations ─────────────────────────────────────
    importances = get_feature_importance(rf_pipeline, feature_names)
    plot_feature_importance(feature_names, importances, "Random Forest")
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)
    plot_risk_segmentation(df, scaler, rf_pipeline)
    plot_model_comparison(results)

    # ── Step 8: Demo prediction ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DEMO PATIENT RISK PREDICTION")
    print("=" * 60)

    # Example 1 — higher-risk profile
    result1 = predict_patient_risk(
        model=rf_pipeline,
        age=62, sex=1, cp=2, trestbps=150, chol=270,
        fbs=1, restecg=1, thalach=125, exang=1,
        oldpeak=2.5, slope=1, ca=2, thal=2,
    )
    print("\n  Patient A (higher-risk profile):")
    print(f"    {result1['interpretation']}")

    # Example 2 — lower-risk profile
    result2 = predict_patient_risk(
        model=rf_pipeline,
        age=38, sex=0, cp=0, trestbps=120, chol=195,
        fbs=0, restecg=0, thalach=170, exang=0,
        oldpeak=0.5, slope=2, ca=0, thal=0,
    )
    print("\n  Patient B (lower-risk profile):")
    print(f"    {result2['interpretation']}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    best_model = max(results, key=lambda k: results[k]["roc_auc"])
    best_auc   = results[best_model]["roc_auc"]
    best_acc   = results[best_model]["accuracy"]
    print(f"  Best Model : {best_model}")
    print(f"  Accuracy   : {best_acc:.4f}")
    print(f"  ROC-AUC    : {best_auc:.4f}")
    print(f"\n  All plots saved to → ./{OUTPUT_DIR}/")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "heart.csv"
    main(filepath)