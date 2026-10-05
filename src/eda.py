"""
eda.py
------
Exploratory Data Analysis for the 9-column customer churn dataset.
Generates and saves professional figures to outputs/figures/.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, FIGURE_DPI, TARGET_COL

COLOR_NO  = "#4C84FF"
COLOR_YES = "#FF4C4C"
PALETTE   = [COLOR_NO, COLOR_YES]

plt.rcParams.update({
    "figure.dpi":      FIGURE_DPI,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
})


def _save(fig, fname):
    p = FIGURES_DIR / fname
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [OK] Saved {fname}")


def run_eda(df: pd.DataFrame) -> None:
    """Generate and save all EDA figures."""
    print("\n  --- EDA: Generating Figures ---")

    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    num_cols = [c for c in ["Tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    cat_cols = [c for c in ["Contract", "PaymentMethod", "PaperlessBilling"] if c in df.columns]

    # ------------------------------------------------------------------
    # 01 Dataset Overview
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("01 - Dataset Overview", fontweight="bold")

    # Table of dimensions
    meta = pd.DataFrame({
        "Metric": ["Total Rows", "Total Columns", "Feature Columns", "Target", "Missing Values"],
        "Value":  [len(df), len(df.columns), len(df.columns) - 2,
                   TARGET_COL, int(df.isnull().sum().sum())]
    })
    axes[0].axis("off")
    tbl = axes[0].table(cellText=meta.values, colLabels=meta.columns,
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.7)
    axes[0].set_title("Dataset Dimensions")

    # Dtype breakdown
    dtype_map = {"int64": "Numeric", "object": "Categorical", "str": "Categorical",
                 "float64": "Numeric", "bool": "Binary"}
    dtype_counts = df.dtypes.astype(str).map(lambda x: dtype_map.get(x, x)).value_counts()
    axes[1].pie(dtype_counts.values,
                labels=[f"{k} ({v})" for k, v in dtype_counts.items()],
                colors=["#4C84FF", "#FF8C42", "#66BB6A"],
                autopct="%1.0f%%", startangle=90)
    axes[1].set_title("Column Types")

    # Missing values
    mv = df.isnull().sum()
    mv = mv[mv > 0]
    if len(mv):
        axes[2].barh(mv.index, mv.values, color=COLOR_YES)
        axes[2].set_title("Missing Values per Column")
    else:
        axes[2].text(0.5, 0.5, "No Missing Values", ha="center",
                     va="center", fontsize=13, color="green")
        axes[2].axis("off")
        axes[2].set_title("Missing Values")

    plt.tight_layout()
    _save(fig, "01_dataset_overview.png")

    # ------------------------------------------------------------------
    # 02 Churn Distribution
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("02 - Churn Distribution", fontweight="bold")

    vc = df[TARGET_COL].value_counts().sort_index()
    labels = ["Not Churned (0)", "Churned (1)"]
    axes[0].pie(vc.values,
                labels=[f"{labels[i]}\n({vc.values[i]})" for i in range(len(vc))],
                colors=PALETTE, autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Class Balance")

    bars = axes[1].bar(["Not Churned", "Churned"], vc.values, color=PALETTE,
                       edgecolor="white", width=0.5)
    for b in bars:
        axes[1].text(b.get_x() + b.get_width() / 2,
                     b.get_height() + 5, str(int(b.get_height())),
                     ha="center", fontweight="bold")
    axes[1].set_title("Churn Count")
    axes[1].set_ylabel("Customers")
    axes[1].set_ylim(0, vc.max() * 1.15)

    plt.tight_layout()
    _save(fig, "02_churn_distribution.png")

    # ------------------------------------------------------------------
    # 03 Numerical Feature Distributions
    # ------------------------------------------------------------------
    if num_cols:
        n = len(num_cols)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
        fig.suptitle("03 - Numerical Feature Distributions", fontweight="bold")
        for i, col in enumerate(num_cols):
            axes[0, i].hist(df[col].dropna(), bins=30,
                            color=COLOR_NO, edgecolor="white", alpha=0.8)
            axes[0, i].set_title(f"{col} (Overall)")
            axes[0, i].set_ylabel("Count")

            for val, color, label in [(0, COLOR_NO, "No Churn"), (1, COLOR_YES, "Churn")]:
                subset = df[df[TARGET_COL] == val][col].dropna()
                axes[1, i].hist(subset, bins=25, alpha=0.65,
                                color=color, edgecolor="white", label=label)
            axes[1, i].set_title(f"{col} by Churn")
            axes[1, i].set_ylabel("Count")
            axes[1, i].legend(fontsize=8)

        plt.tight_layout()
        _save(fig, "03_numerical_distributions.png")

    # ------------------------------------------------------------------
    # 04 Categorical Feature Distributions
    # ------------------------------------------------------------------
    if cat_cols:
        fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 5))
        if len(cat_cols) == 1:
            axes = [axes]
        fig.suptitle("04 - Categorical Feature Distributions", fontweight="bold")
        for i, col in enumerate(cat_cols):
            order = df[col].value_counts().index.tolist()
            sns.countplot(y=col, data=df, order=order,
                          color=COLOR_NO, ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_xlabel("Count")
            axes[i].set_ylabel("")
        plt.tight_layout()
        _save(fig, "04_categorical_distributions.png")

    # ------------------------------------------------------------------
    # 05 Tenure vs Churn
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("05 - Tenure vs Churn", fontweight="bold")

    for val, color, label in [(0, COLOR_NO, "No Churn"), (1, COLOR_YES, "Churn")]:
        axes[0].hist(df[df[TARGET_COL] == val]["Tenure"].dropna(),
                     bins=25, alpha=0.65, color=color, edgecolor="white", label=label)
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Tenure Distribution by Churn")
    axes[0].legend()

    if "TenureGroup" in df.columns:
        order = ["New", "Growing", "Established", "Loyal"]
        order = [o for o in order if o in df["TenureGroup"].values]
        churn_by_group = df.groupby("TenureGroup")[TARGET_COL].mean() * 100
        churn_by_group = churn_by_group.reindex(order).dropna()
        bars2 = axes[1].bar(churn_by_group.index, churn_by_group.values,
                            color=COLOR_YES, edgecolor="white")
        for b in bars2:
            axes[1].text(b.get_x() + b.get_width() / 2,
                         b.get_height() + 0.3,
                         f"{b.get_height():.1f}%", ha="center", fontsize=9)
        axes[1].set_title("Churn Rate (%) by Tenure Group")
        axes[1].set_ylabel("Churn Rate (%)")
        axes[1].set_ylim(0, churn_by_group.max() * 1.25 + 1)

    plt.tight_layout()
    _save(fig, "05_tenure_vs_churn.png")

    # ------------------------------------------------------------------
    # 06 MonthlyCharges vs Churn
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("06 - MonthlyCharges vs Churn", fontweight="bold")

    sns.boxplot(x=TARGET_COL, y="MonthlyCharges", data=df,
                palette=PALETTE, ax=axes[0])
    axes[0].set_xticklabels(["No Churn", "Churn"])
    axes[0].set_title("Monthly Charges Boxplot")

    for val, color, label in [(0, COLOR_NO, "No Churn"), (1, COLOR_YES, "Churn")]:
        axes[1].hist(df[df[TARGET_COL] == val]["MonthlyCharges"].dropna(),
                     bins=25, alpha=0.65, color=color, edgecolor="white", label=label)
    axes[1].set_title("Monthly Charges Distribution")
    axes[1].set_xlabel("Monthly Charges ($)")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, "06_monthly_charges_vs_churn.png")

    # ------------------------------------------------------------------
    # 07 TotalCharges vs Churn
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("07 - TotalCharges vs Churn", fontweight="bold")

    sns.boxplot(x=TARGET_COL, y="TotalCharges", data=df,
                palette=PALETTE, ax=axes[0])
    axes[0].set_xticklabels(["No Churn", "Churn"])
    axes[0].set_title("Total Charges Boxplot")

    for val, color, label in [(0, COLOR_NO, "No Churn"), (1, COLOR_YES, "Churn")]:
        axes[1].hist(df[df[TARGET_COL] == val]["TotalCharges"].dropna(),
                     bins=25, alpha=0.65, color=color, edgecolor="white", label=label)
    axes[1].set_title("Total Charges Distribution")
    axes[1].set_xlabel("Total Charges ($)")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, "07_total_charges_vs_churn.png")

    # ------------------------------------------------------------------
    # 08 Contract vs Churn
    # ------------------------------------------------------------------
    if "Contract" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("08 - Contract Type vs Churn", fontweight="bold")

        ct = df.groupby("Contract")[TARGET_COL].mean() * 100
        ct.sort_values(ascending=False).plot(kind="bar",
            color=COLOR_YES, edgecolor="white", ax=axes[0])
        axes[0].set_title("Churn Rate (%) by Contract")
        axes[0].set_ylabel("Churn Rate (%)")
        axes[0].tick_params(axis="x", rotation=30)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.1f}%",
                             (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
                             ha="center", fontsize=9)

        ct2 = df.groupby("Contract")[TARGET_COL].value_counts().unstack(fill_value=0)
        ct2.plot(kind="bar", color=PALETTE, edgecolor="white", ax=axes[1])
        axes[1].set_title("Count by Contract & Churn")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].legend(["No Churn", "Churn"])

        plt.tight_layout()
        _save(fig, "08_contract_vs_churn.png")

    # ------------------------------------------------------------------
    # 09 PaymentMethod vs Churn
    # ------------------------------------------------------------------
    if "PaymentMethod" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("09 - Payment Method vs Churn", fontweight="bold")

        pm = df.groupby("PaymentMethod")[TARGET_COL].mean() * 100
        pm.sort_values(ascending=False).plot(kind="barh",
            color=COLOR_YES, edgecolor="white", ax=axes[0])
        axes[0].set_title("Churn Rate (%) by Payment Method")
        axes[0].set_xlabel("Churn Rate (%)")

        pm2 = df.groupby("PaymentMethod")[TARGET_COL].value_counts().unstack(fill_value=0)
        pm2.plot(kind="barh", color=PALETTE, edgecolor="white", ax=axes[1])
        axes[1].set_title("Count by Payment Method & Churn")
        axes[1].legend(["No Churn", "Churn"])

        plt.tight_layout()
        _save(fig, "09_payment_method_vs_churn.png")

    # ------------------------------------------------------------------
    # 10 PaperlessBilling vs Churn
    # ------------------------------------------------------------------
    if "PaperlessBilling" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("10 - Paperless Billing vs Churn", fontweight="bold")

        pb = df.groupby("PaperlessBilling")[TARGET_COL].mean() * 100
        pb.plot(kind="bar", color=COLOR_YES, edgecolor="white", ax=axes[0])
        axes[0].set_title("Churn Rate (%) by Paperless Billing")
        axes[0].set_ylabel("Churn Rate (%)")
        axes[0].tick_params(axis="x", rotation=0)

        df.groupby("PaperlessBilling")[TARGET_COL].value_counts().unstack(fill_value=0).plot(
            kind="bar", color=PALETTE, edgecolor="white", ax=axes[1])
        axes[1].set_title("Count by Paperless Billing & Churn")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].legend(["No Churn", "Churn"])

        plt.tight_layout()
        _save(fig, "10_paperless_billing_vs_churn.png")

    # ------------------------------------------------------------------
    # 11 SeniorCitizen vs Churn
    # ------------------------------------------------------------------
    if "SeniorCitizen" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("11 - Senior Citizen vs Churn", fontweight="bold")

        df["SeniorLabel"] = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
        sc = df.groupby("SeniorLabel")[TARGET_COL].mean() * 100
        sc.plot(kind="bar", color=COLOR_YES, edgecolor="white", ax=axes[0])
        axes[0].set_title("Churn Rate (%) by Senior Citizen")
        axes[0].set_ylabel("Churn Rate (%)")
        axes[0].tick_params(axis="x", rotation=0)

        df.groupby("SeniorLabel")[TARGET_COL].value_counts().unstack(fill_value=0).plot(
            kind="bar", color=PALETTE, edgecolor="white", ax=axes[1])
        axes[1].set_title("Count by Senior Citizen & Churn")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].legend(["No Churn", "Churn"])
        df.drop(columns=["SeniorLabel"], inplace=True, errors="ignore")

        plt.tight_layout()
        _save(fig, "11_senior_citizen_vs_churn.png")

    # ------------------------------------------------------------------
    # 12 Correlation Heatmap
    # ------------------------------------------------------------------
    corr_cols = [c for c in
                 ["Tenure", "MonthlyCharges", "TotalCharges",
                  "SeniorCitizen", "ContractRisk", "PaymentRisk",
                  "HighChargeFlag", "EstimatedCLV", TARGET_COL]
                 if c in df.columns]
    if len(corr_cols) >= 3:
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8})
        ax.set_title("12 - Correlation Matrix (Numerical Features + Target)",
                     fontweight="bold")
        plt.tight_layout()
        _save(fig, "12_correlation_matrix.png")

    # ------------------------------------------------------------------
    # 13 Important Relationships — Scatter: Tenure vs MonthlyCharges
    # ------------------------------------------------------------------
    if "Tenure" in df.columns and "MonthlyCharges" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        for val, color, label in [(0, COLOR_NO, "No Churn"), (1, COLOR_YES, "Churn")]:
            subset = df[df[TARGET_COL] == val]
            ax.scatter(subset["Tenure"], subset["MonthlyCharges"],
                       alpha=0.5, c=color, label=label, s=25, edgecolors="none")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Monthly Charges ($)")
        ax.set_title("13 - Tenure vs MonthlyCharges by Churn Status", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        _save(fig, "13_tenure_vs_charges_scatter.png")

    print("  [OK] All EDA figures saved.")
