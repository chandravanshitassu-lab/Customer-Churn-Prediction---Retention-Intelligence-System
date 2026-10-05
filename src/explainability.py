"""
explainability.py  (renamed from old explainability.py — kept for compatibility)
-----------------
Feature importance analysis for the final model.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, TABLES_DIR, FIGURE_DPI, RANDOM_STATE


def get_feature_importance(model, feature_names: list) -> pd.Series | None:
    """Extract importance from tree models or linear coefficients."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_,
                         index=feature_names).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        return pd.Series(np.abs(coef),
                         index=feature_names).sort_values(ascending=False)
    return None


def plot_feature_importance(model, feature_names, X_test=None,
                            y_test=None, top_n=20) -> pd.DataFrame | None:
    """Plot + save feature importance and write CSV."""
    print("\n  --- Feature Importance ---")

    imp = get_feature_importance(model, feature_names)

    if imp is None and X_test is not None:
        print("  No native importances - using permutation importance.")
        result = permutation_importance(model, X_test, y_test,
                                        n_repeats=10, random_state=RANDOM_STATE,
                                        scoring="roc_auc")
        imp = pd.Series(result.importances_mean,
                        index=feature_names).sort_values(ascending=False)

    if imp is None:
        print("  [!] Could not compute feature importance.")
        return None

    top_imp = imp.head(top_n)

    # Save CSV
    imp_df = top_imp.reset_index()
    imp_df.columns = ["Feature", "Importance"]
    csv_path = TABLES_DIR / "feature_importance.csv"
    imp_df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved {csv_path.name}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors = ["#FF4C4C" if i < 5 else "#4C84FF" for i in range(len(top_imp))]
    top_imp[::-1].plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.axvline(x=top_imp.mean(), color="gray", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Mean = {top_imp.mean():.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved feature_importance.png")

    print("\n  Top 10 churn drivers (business interpretation):")
    for rank, (feat, val) in enumerate(top_imp.head(10).items(), 1):
        clean = (feat.replace("num__", "").replace("cat__", "")
                    .replace("_", " ").title())
        print(f"    {rank:2d}. {clean:<42s} {val:.4f}")

    print("\n  NOTE: Importance = correlation with churn, NOT causation.")
    return imp_df
