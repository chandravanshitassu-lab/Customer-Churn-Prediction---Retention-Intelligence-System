"""
src/utils.py
------------
Shared utility functions for the Customer Churn Prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def describe_dataframe(df: pd.DataFrame, title: str = "Dataset") -> None:
    """Print a structured summary of a DataFrame."""
    print(f"\n  === {title} ===")
    print(f"  Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Missing : {df.isnull().sum().sum()} total")
    print(f"  Dtypes  :")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<30s}: {dtype}")


def summarise_target(df: pd.DataFrame, target_col: str = "Churn") -> None:
    """Print class distribution for the target column."""
    if target_col not in df.columns:
        print(f"  [!] Column '{target_col}' not found.")
        return
    vc = df[target_col].value_counts()
    total = len(df)
    print(f"\n  Target Column: {target_col}")
    for val, count in sorted(vc.items()):
        print(f"    {val} : {count:>5d} ({count/total*100:.1f}%)")


def check_class_imbalance(df: pd.DataFrame, target_col: str = "Churn") -> str:
    """
    Return imbalance severity as a string label.

    Returns
    -------
    str : 'Balanced', 'Mild Imbalance', 'Moderate Imbalance', 'Severe Imbalance'
    """
    minority_rate = df[target_col].mean()
    if minority_rate >= 0.35:
        return "Balanced"
    elif minority_rate >= 0.20:
        return "Mild Imbalance"
    elif minority_rate >= 0.10:
        return "Moderate Imbalance"
    else:
        return "Severe Imbalance"


def safe_divide(a, b, fill=0.0):
    """Divide a by b; return fill if b == 0."""
    if b == 0:
        return fill
    return a / b


def format_metrics_table(metrics_dict: dict) -> str:
    """Format a metrics dict as a readable table string."""
    lines = []
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            lines.append(f"  {k:<20s}: {v:.4f}")
        else:
            lines.append(f"  {k:<20s}: {v}")
    return "\n".join(lines)
