# model_training.py
# Logistic Regression, XGBoost and LightGBM for malaria and TB prediction
# ZENIK.AI — Computational Epidemiology Portfolio

import sys
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.patient_data import generate_malaria_dataset, generate_tb_dataset
from preprocessing import preprocess_malaria, preprocess_tb

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name, disease):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{model_name} — {disease}")
    print(f"AUC-ROC: {auc:.3f}")
    print(classification_report(y_test, y_pred,
          target_names=["No event", "Event"]))

    return auc, y_prob


def plot_roc_curves(y_test, probs_dict, disease, filename):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — {disease} Outcome Prediction\nKenya Patient Data")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved: {path}")


def train_malaria_models(X_train, X_test, y_train, y_test, features):
    """Train and evaluate all models for malaria prediction."""
    print("\n" + "="*60)
    print("MALARIA SEVERE OUTCOME PREDICTION")
    print("="*60)

    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Logistic Regression — interpretable baseline
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_sc, y_train)
    auc_lr, prob_lr = evaluate_model(lr, X_test_sc, y_test,
                                      "Logistic Regression", "Malaria")

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    auc_xgb, prob_xgb = evaluate_model(xgb_model, X_test, y_test,
                                        "XGBoost", "Malaria")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    auc_lgb, prob_lgb = evaluate_model(lgb_model, X_test, y_test,
                                        "LightGBM", "Malaria")

    plot_roc_curves(y_test,
                    {"Logistic Regression": prob_lr,
                     "XGBoost": prob_xgb,
                     "LightGBM": prob_lgb},
                    "Malaria", "malaria_roc_curves.png")

    # Save best model
    best_model = xgb_model if auc_xgb >= auc_lgb else lgb_model
    with open(os.path.join(OUTPUT_DIR, "malaria_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler,
                     "features": features}, f)
    print(f"Best malaria model saved.")

    return best_model, scaler, xgb_model


def train_tb_models(X_train, X_test, y_train, y_test, features):
    """Train and evaluate all models for TB treatment failure prediction."""
    print("\n" + "="*60)
    print("TB TREATMENT FAILURE PREDICTION")
    print("="*60)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_sc, y_train)
    auc_lr, prob_lr = evaluate_model(lr, X_test_sc, y_test,
                                      "Logistic Regression", "TB")

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    auc_xgb, prob_xgb = evaluate_model(xgb_model, X_test, y_test,
                                        "XGBoost", "TB")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    auc_lgb, prob_lgb = evaluate_model(lgb_model, X_test, y_test,
                                        "LightGBM", "TB")

    plot_roc_curves(y_test,
                    {"Logistic Regression": prob_lr,
                     "XGBoost": prob_xgb,
                     "LightGBM": prob_lgb},
                    "TB", "tb_roc_curves.png")

    best_model = xgb_model if auc_xgb >= auc_lgb else lgb_model
    with open(os.path.join(OUTPUT_DIR, "tb_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler,
                     "features": features}, f)
    print(f"Best TB model saved.")

    return best_model, scaler, xgb_model


if __name__ == "__main__":
    malaria_df = generate_malaria_dataset(1000)
    tb_df = generate_tb_dataset(800)

    X_train_m, X_test_m, y_train_m, y_test_m, features_m = preprocess_malaria(malaria_df)
    X_train_t, X_test_t, y_train_t, y_test_t, features_t = preprocess_tb(tb_df)

    malaria_model, malaria_scaler, malaria_xgb = train_malaria_models(
        X_train_m, X_test_m, y_train_m, y_test_m, features_m)

    tb_model, tb_scaler, tb_xgb = train_tb_models(
        X_train_t, X_test_t, y_train_t, y_test_t, features_t)

    print("\nModel training complete.")
    print("ROC curves saved to outputs folder.")