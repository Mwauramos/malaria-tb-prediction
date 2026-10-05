# shap_explainability.py
# SHAP waterfall and summary plots for malaria and TB models
# ZENIK.AI — Computational Epidemiology Portfolio

import sys
import os
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.patient_data import generate_malaria_dataset, generate_tb_dataset
from preprocessing import preprocess_malaria, preprocess_tb
from model_training import train_malaria_models, train_tb_models

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def explain_malaria_model(model, X_test, features):
    """Generate SHAP plots for malaria model."""
    print("\nGenerating SHAP explanations for malaria model...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot — global feature importance
    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=features,
        show=False,
        plot_size=(10, 6)
    )
    plt.title("SHAP Feature Importance — Severe Malaria Prediction\nKenya Patient Data")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "malaria_shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved: {path}")

    # Waterfall plot for one high-risk patient
    high_risk_idx = np.argmax(shap_values.sum(axis=1) if isinstance(shap_values, np.ndarray) else shap_values)

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[high_risk_idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[high_risk_idx].values,
            feature_names=features
        ),
        show=False
    )
    plt.title("SHAP Waterfall — Highest Risk Malaria Patient")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "malaria_shap_waterfall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP waterfall plot saved: {path}")


def explain_tb_model(model, X_test, features):
    """Generate SHAP plots for TB model."""
    print("\nGenerating SHAP explanations for TB model...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=features,
        show=False,
        plot_size=(10, 6)
    )
    plt.title("SHAP Feature Importance — TB Treatment Failure Prediction\nKenya Patient Data")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "tb_shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved: {path}")

    high_risk_idx = np.argmax(shap_values.sum(axis=1) if isinstance(shap_values, np.ndarray) else shap_values)

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[high_risk_idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[high_risk_idx].values,
            feature_names=features
        ),
        show=False
    )
    plt.title("SHAP Waterfall — Highest Risk TB Patient")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "tb_shap_waterfall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP waterfall plot saved: {path}")


if __name__ == "__main__":
    malaria_df = generate_malaria_dataset(1000)
    tb_df = generate_tb_dataset(800)

    X_train_m, X_test_m, y_train_m, y_test_m, features_m = preprocess_malaria(malaria_df)
    X_train_t, X_test_t, y_train_t, y_test_t, features_t = preprocess_tb(tb_df)

    malaria_model, malaria_scaler, malaria_xgb = train_malaria_models(
        X_train_m, X_test_m, y_train_m, y_test_m, features_m)
    tb_model, tb_scaler, tb_xgb = train_tb_models(
        X_train_t, X_test_t, y_train_t, y_test_t, features_t)

    explain_malaria_model(malaria_xgb, X_test_m, features_m)
    explain_tb_model(tb_xgb, X_test_t, features_t)

    print("\nSHAP analysis complete. Check outputs folder for plots.")