# preprocessing.py
# Data cleaning, feature engineering and train/test split
# ZENIK.AI — Computational Epidemiology Portfolio

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.patient_data import generate_malaria_dataset, generate_tb_dataset


def preprocess_malaria(df: pd.DataFrame):
    """Prepare malaria dataset for modelling."""
    print("Preprocessing malaria dataset...")

    features = [
        "age", "iron_deficiency", "bed_net_use", "hiv_positive",
        "prior_malaria_episode", "haemoglobin_gdl", "parasite_density",
        "days_to_treatment", "wealth_quintile"
    ]

    # Encode sex
    df["sex_male"] = (df["sex"] == "Male").astype(int)
    features.append("sex_male")

    # Encode county malaria risk zone
    high_risk_counties = ["Kisumu", "Homa Bay", "Migori", "Siaya"]
    df["high_risk_county"] = df["county"].isin(high_risk_counties).astype(int)
    features.append("high_risk_county")

    X = df[features]
    y = df["severe_malaria"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"Training samples after SMOTE: {len(X_train_bal)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {features}")

    return X_train_bal, X_test, y_train_bal, y_test, features


def preprocess_tb(df: pd.DataFrame):
    """Prepare TB dataset for modelling."""
    print("\nPreprocessing TB dataset...")

    features = [
        "age", "hiv_positive", "mdr_tb", "dot_therapy",
        "diabetes", "smoking", "bmi", "sputum_grade",
        "household_contacts", "wealth_quintile"
    ]

    df["sex_male"] = (df["sex"] == "Male").astype(int)
    features.append("sex_male")

    urban_counties = ["Nairobi", "Mombasa", "Nakuru"]
    df["urban_county"] = df["county"].isin(urban_counties).astype(int)
    features.append("urban_county")

    X = df[features]
    y = df["treatment_failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"Training samples after SMOTE: {len(X_train_bal)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {features}")

    return X_train_bal, X_test, y_train_bal, y_test, features


if __name__ == "__main__":
    malaria_df = generate_malaria_dataset(1000)
    tb_df = generate_tb_dataset(800)

    X_train_m, X_test_m, y_train_m, y_test_m, features_m = preprocess_malaria(malaria_df)
    X_train_t, X_test_t, y_train_t, y_test_t, features_t = preprocess_tb(tb_df)

    print("\nPreprocessing complete.")
    print(f"Malaria class balance after SMOTE: {y_train_m.value_counts().to_dict()}")
    print(f"TB class balance after SMOTE: {y_train_t.value_counts().to_dict()}")