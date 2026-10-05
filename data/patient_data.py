# patient_data.py
# Synthetic Kenya malaria and TB patient dataset
# Based on real epidemiological distributions from KEMRI and Kenya DHIS2
# ZENIK.AI — Computational Epidemiology Portfolio

import numpy as np
import pandas as pd

np.random.seed(42)

KENYA_COUNTIES = [
    "Kisumu", "Homa Bay", "Migori", "Siaya",      # Lake endemic zone — high malaria
    "Nairobi", "Mombasa", "Nakuru", "Kisii",       # Urban — high TB
    "Turkana", "Marsabit", "Wajir", "Garissa",     # Arid — mixed burden
    "Kakamega", "Bungoma", "Vihiga", "Trans Nzoia" # Western — moderate burden
]

COUNTY_MALARIA_RISK = {
    "Kisumu": 0.75, "Homa Bay": 0.80, "Migori": 0.72, "Siaya": 0.78,
    "Nairobi": 0.15, "Mombasa": 0.35, "Nakuru": 0.20, "Kisii": 0.45,
    "Turkana": 0.40, "Marsabit": 0.38, "Wajir": 0.35, "Garissa": 0.42,
    "Kakamega": 0.55, "Bungoma": 0.50, "Vihiga": 0.52, "Trans Nzoia": 0.48
}


def generate_malaria_dataset(n=1000):
    """Generate synthetic malaria patient data for Kenya."""
    counties = np.random.choice(KENYA_COUNTIES, size=n)
    ages = np.random.gamma(shape=2, scale=10, size=n).clip(0, 85).astype(int)
    sex = np.random.choice(["Male", "Female"], size=n, p=[0.51, 0.49])
    
    # Clinical features based on real malaria risk factors
    iron_deficiency = np.random.binomial(1, 0.45, n)
    bed_net_use = np.random.binomial(1, 0.55, n)
    hiv_status = np.random.binomial(1, 0.06, n)
    prior_malaria = np.random.binomial(1, 0.60, n)
    haemoglobin = np.random.normal(10.5, 2.5, n).clip(4, 18)
    parasite_density = np.random.exponential(scale=5000, size=n).clip(100, 100000)
    days_to_treatment = np.random.exponential(scale=2, size=n).clip(0, 14).astype(int)
    wealth_quintile = np.random.choice([1, 2, 3, 4, 5], size=n,
                                        p=[0.25, 0.25, 0.20, 0.18, 0.12])

    # Outcome: severe malaria (based on real risk factors)
    county_risk = np.array([COUNTY_MALARIA_RISK[c] for c in counties])
    log_odds = (
        -2.0
        + 0.8 * (ages < 5).astype(int)
        + 0.6 * iron_deficiency
        - 0.5 * bed_net_use
        + 0.7 * hiv_status
        + 0.4 * prior_malaria
        - 0.3 * (haemoglobin > 11).astype(int)
        + 0.5 * np.log(parasite_density / 1000)
        + 0.4 * days_to_treatment
        - 0.3 * (wealth_quintile >= 4).astype(int)
        + np.log(county_risk / (1 - county_risk))
    )
    prob_severe = 1 / (1 + np.exp(-log_odds))
    severe_malaria = np.random.binomial(1, prob_severe)

    df = pd.DataFrame({
        "patient_id": [f"MAL{i:04d}" for i in range(n)],
        "county": counties,
        "age": ages,
        "age_group": pd.cut(ages, bins=[0,5,15,25,45,60,85],
                            labels=["Under 5","5 to 14","15 to 24",
                                    "25 to 44","45 to 59","60 plus"]),
        "sex": sex,
        "iron_deficiency": iron_deficiency,
        "bed_net_use": bed_net_use,
        "hiv_positive": hiv_status,
        "prior_malaria_episode": prior_malaria,
        "haemoglobin_gdl": haemoglobin.round(1),
        "parasite_density": parasite_density.round(0).astype(int),
        "days_to_treatment": days_to_treatment,
        "wealth_quintile": wealth_quintile,
        "severe_malaria": severe_malaria,
        "disease": "Malaria"
    })
    return df


def generate_tb_dataset(n=800):
    """Generate synthetic TB patient data for Kenya."""
    counties = np.random.choice(KENYA_COUNTIES, size=n)
    ages = np.random.normal(35, 15, size=n).clip(15, 80).astype(int)
    sex = np.random.choice(["Male", "Female"], size=n, p=[0.62, 0.38])

    hiv_status = np.random.binomial(1, 0.35, n)
    mdr_tb = np.random.binomial(1, 0.04, n)
    dot_therapy = np.random.binomial(1, 0.70, n)
    diabetes = np.random.binomial(1, 0.08, n)
    smoking = np.random.binomial(1, 0.22, n)
    bmi = np.random.normal(19.5, 3.5, n).clip(13, 35)
    sputum_grade = np.random.choice([1, 2, 3], size=n, p=[0.40, 0.35, 0.25])
    household_contacts = np.random.poisson(3, n).clip(0, 10)
    wealth_quintile = np.random.choice([1, 2, 3, 4, 5], size=n,
                                        p=[0.30, 0.28, 0.20, 0.14, 0.08])

    # Outcome: treatment failure
    log_odds = (
        -1.5
        + 1.2 * hiv_status
        + 1.5 * mdr_tb
        - 0.8 * dot_therapy
        + 0.5 * diabetes
        + 0.4 * smoking
        - 0.4 * (bmi > 18.5).astype(int)
        + 0.3 * (sputum_grade == 3).astype(int)
        - 0.3 * (wealth_quintile >= 4).astype(int)
    )
    prob_failure = 1 / (1 + np.exp(-log_odds))
    treatment_failure = np.random.binomial(1, prob_failure)

    df = pd.DataFrame({
        "patient_id": [f"TB{i:04d}" for i in range(n)],
        "county": counties,
        "age": ages,
        "age_group": pd.cut(ages, bins=[15,25,35,45,55,65,80],
                            labels=["15 to 24","25 to 34","35 to 44",
                                    "45 to 54","55 to 64","65 plus"]),
        "sex": sex,
        "hiv_positive": hiv_status,
        "mdr_tb": mdr_tb,
        "dot_therapy": dot_therapy,
        "diabetes": diabetes,
        "smoking": smoking,
        "bmi": bmi.round(1),
        "sputum_grade": sputum_grade,
        "household_contacts": household_contacts,
        "wealth_quintile": wealth_quintile,
        "treatment_failure": treatment_failure,
        "disease": "TB"
    })
    return df


if __name__ == "__main__":
    malaria_df = generate_malaria_dataset(1000)
    tb_df = generate_tb_dataset(800)

    print("MALARIA DATASET")
    print(f"Patients: {len(malaria_df)}")
    print(f"Severe malaria rate: {malaria_df['severe_malaria'].mean():.1%}")
    print(f"Counties: {malaria_df['county'].nunique()}")
    print(malaria_df.head(3))

    print("\nTB DATASET")
    print(f"Patients: {len(tb_df)}")
    print(f"Treatment failure rate: {tb_df['treatment_failure'].mean():.1%}")
    print(malaria_df[['age_group','sex','iron_deficiency',
                       'bed_net_use','severe_malaria']].describe())

    print("\nDatasets generated successfully.")