"""
src/clustering.py
-----------------
Customer segmentation using K-Means clustering.

While the primary task is churn prediction (classification),
clustering provides additional business intelligence by revealing
natural customer segments that can inform retention strategy.

Note: Clustering is exploratory and unsupervised.
      The Churn label is NOT used during clustering (no leakage).
      It is only examined AFTER clustering to understand segment profiles.
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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, TABLES_DIR, FIGURE_DPI, RANDOM_STATE


def run_clustering(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Perform K-Means clustering on numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned churn dataset.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    pd.DataFrame
        Original df with 'Cluster' column added.
    """
    print("\n  --- Customer Segmentation (K-Means) ---")

    # Use numerical features only (no leakage — drop Churn before clustering)
    cluster_features = [c for c in
                        ["Tenure", "MonthlyCharges", "TotalCharges",
                         "SeniorCitizen", "EstimatedCLV", "AverageMonthlySpend"]
                        if c in df.columns]

    X = df[cluster_features].copy()

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k with silhouette score (k=2..5)
    best_k, best_sil = n_clusters, -1
    sil_scores = {}
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores[k] = round(sil, 4)
        if sil > best_sil:
            best_k, best_sil = k, sil

    print(f"  Silhouette scores: {sil_scores}")
    print(f"  Best k = {best_k} (silhouette = {best_sil:.4f})")

    # Fit final model
    kmeans  = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels  = kmeans.fit_predict(X_scaled)
    df      = df.copy()
    df["Cluster"] = labels

    # Cluster profiles
    profile_cols = cluster_features + (
        ["Churn"] if "Churn" in df.columns else []
    )
    profile = df.groupby("Cluster")[profile_cols].mean().round(2)
    print("\n  Cluster Profiles:")
    print(profile.to_string())

    out_path = TABLES_DIR / "cluster_profiles.csv"
    profile.to_csv(out_path)
    print(f"  [OK] Cluster profiles saved -> {out_path.name}")

    # PCA 2D visualisation
    pca      = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca    = pca.fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Customer Segmentation (K-Means Clustering)", fontweight="bold")

    # Clusters
    colors = ["#4C84FF", "#FF4C4C", "#4CAF50", "#FF8C42"]
    for c in range(best_k):
        mask = labels == c
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors[c % len(colors)], alpha=0.5,
                        s=20, label=f"Cluster {c}")
    axes[0].set_title("Clusters (PCA 2D)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend()

    # Churn overlay
    if "Churn" in df.columns:
        churn_colors = ["#4C84FF", "#FF4C4C"]
        for val, color, label in [(0, "#4C84FF", "No Churn"), (1, "#FF4C4C", "Churn")]:
            mask = df["Churn"].values == val
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=color, alpha=0.5, s=20, label=label)
        axes[1].set_title("Churn Overlay (PCA 2D)")
        axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
        axes[1].legend()

    plt.tight_layout()
    fig_path = FIGURES_DIR / "customer_segmentation.png"
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Segmentation figure saved -> {fig_path.name}")

    return df
