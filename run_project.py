"""
run_project.py
--------------
Master end-to-end pipeline for the Customer Churn Prediction project.

Usage:
    python run_project.py

Steps:
  1.  Load actual provided datasets
  2.  Validate data
  3.  Clean data
  4.  Feature engineering
  5.  EDA (figures)
  6.  Preprocessing + train/test split
  7.  Train all models
  8.  Evaluate all models
  9.  Cross-validate
  10. Hyperparameter tuning
  11. Select + save final model
  12. Feature importance
  13. Customer risk scoring
  14. Business recommendations
"""

import sys, time, warnings, joblib
warnings.filterwarnings("ignore")

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader        import load_churn_data
from src.data_validation    import run_validation
from src.data_cleaning      import clean_data
from src.feature_engineering import engineer_features, apply_charge_features
from src.eda                import run_eda
from src.preprocessing      import split_and_preprocess
from src.modeling           import (
    train_all_models, cross_validate_models,
    tune_model, save_best_model,
)
from src.evaluation         import (
    evaluate_all_models, append_model_evaluation, plot_confusion_matrices,
    plot_roc_curves, plot_precision_recall_curves,
    plot_model_comparison, select_best_model,
)
from src.explainability     import plot_feature_importance
from src.business_insights  import generate_predictions, print_business_recommendations
from src.config             import TARGET_COL, ID_COL, FEATURE_THRESHOLDS_PKL


def banner(text: str) -> None:
    sep = "=" * 66
    print(f"\n{sep}\n  {text}\n{sep}")


def main() -> None:
    t0 = time.time()

    # Step 1 — Load data
    banner("STEP 1 - DATA LOADING")
    df_raw = load_churn_data()
    print(f"  Dataset shape : {df_raw.shape}")
    print(f"  Churn rate    : {df_raw[TARGET_COL].mean()*100:.1f}%")

    # Step 2 — Validate
    banner("STEP 2 - DATA VALIDATION")
    run_validation(df_raw)

    # Step 3 — Clean
    banner("STEP 3 - DATA CLEANING")
    df_clean = clean_data(df_raw)

    # Step 4 — Feature engineering
    banner("STEP 4 - FEATURE ENGINEERING")
    df_feat = engineer_features(df_clean)

    # Step 5 — EDA
    banner("STEP 5 - EXPLORATORY DATA ANALYSIS")
    run_eda(df_feat)

    # Step 6 — Preprocessing
    banner("STEP 6 - PREPROCESSING & TRAIN/TEST SPLIT")
    (
        X_train, X_test,
        y_train, y_test,
        preprocessor,
        feature_names,
        X_train_raw, X_test_raw,
    ) = split_and_preprocess(df_feat)

    # Step 7 — Train
    banner("STEP 7 - MODEL TRAINING")
    models = train_all_models(X_train, y_train)

    # Step 8 — Evaluate baselines on held-out test
    banner("STEP 8 - MODEL EVALUATION")
    comp_df = evaluate_all_models(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_precision_recall_curves(models, X_test, y_test)
    plot_model_comparison(comp_df)

    # Step 9 — Cross-validate (training data only)
    banner("STEP 9 - CROSS-VALIDATION")
    cv_df = cross_validate_models(models, X_train, y_train)

    # Step 10 — Hyperparameter tuning  (tune Random Forest on train CV)
    banner("STEP 10 - HYPERPARAMETER TUNING")
    tuned_model, _, tuned_cv = tune_model(
        X_train, y_train, model_name="Random Forest", n_iter=20
    )

    banner("STEP 10b - TUNED MODEL HELD-OUT TEST EVALUATION")
    tuned_name = "Random Forest (Tuned)"
    comp_df = append_model_evaluation(
        comp_df, tuned_name, tuned_model, X_test, y_test
    )
    models_with_tuned = dict(models)
    models_with_tuned[tuned_name] = tuned_model
    plot_confusion_matrices(models_with_tuned, X_test, y_test)
    plot_roc_curves(models_with_tuned, X_test, y_test)
    plot_precision_recall_curves(models_with_tuned, X_test, y_test)
    plot_model_comparison(comp_df)

    # Step 11 — Select + save final model (CV, not test)
    banner("STEP 11 - FINAL MODEL SELECTION")
    final_model, final_name = select_best_model(
        models, comp_df,
        cv_df=cv_df,
        tuned_model=tuned_model,
        tuned_name=tuned_name,
        tuned_cv=tuned_cv,
    )
    save_best_model(final_model, final_name)

    # Step 12 — Feature importance
    banner("STEP 12 - FEATURE IMPORTANCE")
    plot_feature_importance(final_model, feature_names, X_test, y_test)

    # Step 13 — Customer risk scoring (train-only charge cuts)
    banner("STEP 13 - CUSTOMER RISK SCORING")
    charge_thresholds = joblib.load(FEATURE_THRESHOLDS_PKL)
    df_scored = apply_charge_features(df_feat, charge_thresholds)
    drop_cols = [c for c in [ID_COL, TARGET_COL] if c in df_scored.columns]
    X_all_raw = df_scored.drop(columns=drop_cols)
    predictions_df = generate_predictions(
        final_model, preprocessor, df_scored, X_all_raw
    )

    # Step 14 — Business insights
    banner("STEP 14 - BUSINESS RECOMMENDATIONS")
    print_business_recommendations(predictions_df, final_name, comp_df)

    # Done
    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE - {elapsed:.1f}s")

    from src.config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, TABLES_DIR
    n_figs = len(list(FIGURES_DIR.glob("*.png")))
    print(f"""
  Outputs generated:
  +-- outputs/figures/      {n_figs} figures
  +-- outputs/metrics/      model_comparison.csv, cv_results.csv, hp_results.csv
  +-- outputs/predictions/  customer_risk_predictions.csv
  +-- outputs/tables/       data_quality_report.csv, feature_importance.csv
  +-- models/               best_model.pkl, preprocessing_pipeline.pkl
  +-- data/processed/       processed_customer_churn.csv
""")


if __name__ == "__main__":
    main()
