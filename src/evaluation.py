"""
evaluation.py
-------------
Compute metrics and generate all evaluation plots.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, METRICS_DIR, TABLES_DIR, FIGURE_DPI


def evaluate_model(name, model, X_test, y_test) -> dict:
    """Compute all metrics for a single model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),              4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
        "ROC_AUC":   round(roc_auc_score(y_test, y_prob),               4),
        "PR_AUC":    round(average_precision_score(y_test, y_prob),     4),
    }


def save_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by ROC-AUC and persist held-out test metrics."""
    df = df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    out = METRICS_DIR / "model_comparison.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n  [OK] Model comparison saved -> {out.name}")
    return df


def evaluate_all_models(models, X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and save comparison CSV."""
    print("\n  --- Model Evaluation ---")
    rows = [evaluate_model(n, m, X_test, y_test) for n, m in models.items()]
    df   = pd.DataFrame(rows)

    best_name  = df.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
    best_model = models[best_name]
    y_pred     = best_model.predict(X_test)
    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred,
                                target_names=["No Churn", "Churn"],
                                zero_division=0))

    return save_model_comparison(df)


def append_model_evaluation(comp_df, name, model, X_test, y_test) -> pd.DataFrame:
    """Add one extra model's held-out test metrics (e.g. tuned RF)."""
    print(f"\n  --- Held-out Test Evaluation: {name} ---")
    row = evaluate_model(name, model, X_test, y_test)
    print(
        f"  Accuracy={row['Accuracy']:.4f}  Precision={row['Precision']:.4f}  "
        f"Recall={row['Recall']:.4f}  F1={row['F1']:.4f}  "
        f"ROC-AUC={row['ROC_AUC']:.4f}  PR-AUC={row['PR_AUC']:.4f}"
    )
    y_pred = model.predict(X_test)
    print(f"\n  Classification Report ({name}):")
    print(classification_report(y_test, y_pred,
                                target_names=["No Churn", "Churn"],
                                zero_division=0))
    df = pd.concat([comp_df, pd.DataFrame([row])], ignore_index=True)
    return save_model_comparison(df)


def plot_confusion_matrices(models, X_test, y_test) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", fontweight="bold")
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"])
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    p = FIGURES_DIR / "confusion_matrices.png"
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("    [OK] Saved confusion_matrices.png")


def plot_roc_curves(models, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C84FF", "#FF4C4C", "#4CAF50", "#FF8C42", "#9C27B0"]
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    p = FIGURES_DIR / "roc_curves.png"
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("    [OK] Saved roc_curves.png")


def plot_precision_recall_curves(models, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C84FF", "#FF4C4C", "#4CAF50", "#FF8C42", "#9C27B0"]
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})",
                color=color, linewidth=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    p = FIGURES_DIR / "precision_recall_curves.png"
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("    [OK] Saved precision_recall_curves.png")


def plot_model_comparison(comp_df: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    metrics = [m for m in metrics if m in comp_df.columns]
    colors  = ["#4C84FF", "#4CAF50", "#FF4C4C", "#FF8C42", "#9C27B0"]
    x       = np.arange(len(comp_df))
    width   = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics) / 2) * width + width / 2
        ax.bar(x + offset, comp_df[metric], width, label=metric,
               color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.2)
    ax.set_title("Model Comparison — Key Metrics", fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    plt.tight_layout()
    p = FIGURES_DIR / "model_comparison_chart.png"
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("    [OK] Saved model_comparison_chart.png")


def select_best_model(models, comp_df, cv_df=None, tuned_model=None,
                      tuned_name="", tuned_cv=None) -> tuple:
    """
    Select the production model using train-only cross-validation.

    Held-out test metrics in comp_df are reported, not used for selection.
    The tuned candidate is included via its RandomizedSearchCV mean AUC.
    """
    print(f"\n  --- Model Selection (by 5-fold CV ROC-AUC) ---")
    candidates = []
    if cv_df is not None and len(cv_df):
        for _, row in cv_df.iterrows():
            name = row["Model"]
            if name in models:
                candidates.append((float(row["CV_Mean_AUC"]), name, models[name]))
                print(f"  {name:<30s}  CV AUC {row['CV_Mean_AUC']:.4f}")
    if tuned_model is not None and tuned_cv is not None:
        candidates.append((float(tuned_cv), tuned_name, tuned_model))
        print(f"  {tuned_name:<30s}  CV AUC {float(tuned_cv):.4f}")

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        score, name, model = candidates[0]
        print(f"  Final selected : {name} (CV AUC {score:.4f})")
        return model, name

    best_row  = comp_df.iloc[0]
    best_name = best_row["Model"]
    print(f"  Fallback (test ROC-AUC): {best_name}")
    return models[best_name], best_name
