# fairness_audit.py
# Fairness audit across demographic subgroups for malaria and TB models
# ZENIK.AI — Computational Epidemiology Portfolio

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.patient_data import generate_malaria_dataset, generate_tb_dataset
from preprocessing import preprocess_malaria, preprocess_tb
from model_training import train_malaria_models, train_tb_models

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def audit_by_subgroup(model, X_test, y_test, df_test, group_col, label, disease):
    """Compute performance metrics per subgroup."""
    results = []
    groups = df_test[group_col].unique()

    for group in sorted(groups, key=str):
        mask = df_test[group_col] == group
        if mask.sum() < 10:
            continue

        X_sub = X_test[mask]
        y_sub = y_test[mask]

        if y_sub.nunique() < 2:
            continue

        y_prob = model.predict_proba(X_sub)[:, 1]
        y_pred = model.predict(X_sub)

        auc = roc_auc_score(y_sub, y_prob)
        recall = recall_score(y_sub, y_pred, zero_division=0)
        precision = precision_score(y_sub, y_pred, zero_division=0)
        n = mask.sum()
        event_rate = y_sub.mean()

        results.append({
            "subgroup": str(group),
            "n": n,
            "event_rate": round(event_rate, 3),
            "auc": round(auc, 3),
            "recall": round(recall, 3),
            "precision": round(precision, 3)
        })

    df_results = pd.DataFrame(results)
    print(f"\nFairness Audit — {disease} by {label}")
    print(df_results.to_string(index=False))
    return df_results


def plot_fairness(df_results, group_col, metric, disease, filename):
    """Bar chart of fairness metric across subgroups."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if v >= 0.6 else "#e74c3c" for v in df_results[metric]]
    bars = ax.bar(df_results["subgroup"], df_results[metric], color=colors)
    ax.axhline(y=0.6, color="navy", linestyle="--", linewidth=1.5,
               label="Acceptable threshold (0.60)")
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{disease} Model — {metric.upper()} by {group_col}\n"
                 f"Green = acceptable, Red = below threshold")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fairness plot saved: {path}")


if __name__ == "__main__":
    malaria_df = generate_malaria_dataset(1000)
    tb_df = generate_tb_dataset(800)

    X_train_m, X_test_m, y_train_m, y_test_m, features_m = preprocess_malaria(malaria_df)
    X_train_t, X_test_t, y_train_t, y_test_t, features_t = preprocess_tb(tb_df)

    malaria_model, _, malaria_xgb = train_malaria_models(
        X_train_m, X_test_m, y_train_m, y_test_m, features_m)
    tb_model, _, tb_xgb = train_tb_models(
        X_train_t, X_test_t, y_train_t, y_test_t, features_t)

    # Reconstruct test sets with demographic info
    from sklearn.model_selection import train_test_split
    _, malaria_test_df = train_test_split(malaria_df, test_size=0.2,
                                           random_state=42,
                                           stratify=malaria_df["severe_malaria"])
    _, tb_test_df = train_test_split(tb_df, test_size=0.2,
                                      random_state=42,
                                      stratify=tb_df["treatment_failure"])

    malaria_test_df = malaria_test_df.reset_index(drop=True)
    tb_test_df = tb_test_df.reset_index(drop=True)
    X_test_m = X_test_m.reset_index(drop=True)
    X_test_t = X_test_t.reset_index(drop=True)
    y_test_m = y_test_m.reset_index(drop=True)
    y_test_t = y_test_t.reset_index(drop=True)

    # Malaria fairness by sex and age group
    sex_audit_m = audit_by_subgroup(malaria_xgb, X_test_m, y_test_m,
                                     malaria_test_df, "sex",
                                     "Sex", "Malaria")
    plot_fairness(sex_audit_m, "Sex", "auc", "Malaria",
                  "malaria_fairness_sex.png")

    age_audit_m = audit_by_subgroup(malaria_xgb, X_test_m, y_test_m,
                                     malaria_test_df, "age_group",
                                     "Age Group", "Malaria")
    plot_fairness(age_audit_m, "Age Group", "auc", "Malaria",
                  "malaria_fairness_age.png")

    # TB fairness by sex and age group
    sex_audit_t = audit_by_subgroup(tb_xgb, X_test_t, y_test_t,
                                     tb_test_df, "sex",
                                     "Sex", "TB")
    plot_fairness(sex_audit_t, "Sex", "auc", "TB",
                  "tb_fairness_sex.png")

    age_audit_t = audit_by_subgroup(tb_xgb, X_test_t, y_test_t,
                                     tb_test_df, "age_group",
                                     "Age Group", "TB")
    plot_fairness(age_audit_t, "Age Group", "auc", "TB",
                  "tb_fairness_age.png")

    print("\nFairness audit complete. Check outputs folder.")