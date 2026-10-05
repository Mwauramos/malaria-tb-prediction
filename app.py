# app.py
# Streamlit UI for Malaria and TB Outcome Prediction
# ZENIK.AI — Computational Epidemiology Portfolio

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

st.set_page_config(
    page_title="Kenya Malaria & TB Outcome Predictor",
    page_icon="🦟",
    layout="wide"
)

st.title("🦟 Kenya Malaria & TB Outcome Predictor")
st.markdown(
    "Clinical prediction tool for severe malaria and TB treatment failure "
    "in Kenya. Built using XGBoost with SHAP explainability and demographic "
    "fairness audit."
)
st.divider()

@st.cache_resource
def load_models():
    from data.patient_data import generate_malaria_dataset, generate_tb_dataset
    from preprocessing import preprocess_malaria, preprocess_tb
    from model_training import train_malaria_models, train_tb_models

    malaria_df = generate_malaria_dataset(1000)
    tb_df = generate_tb_dataset(800)

    X_train_m, X_test_m, y_train_m, y_test_m, features_m = preprocess_malaria(malaria_df)
    X_train_t, X_test_t, y_train_t, y_test_t, features_t = preprocess_tb(tb_df)

    _, _, malaria_xgb = train_malaria_models(
        X_train_m, X_test_m, y_train_m, y_test_m, features_m)
    _, _, tb_xgb = train_tb_models(
        X_train_t, X_test_t, y_train_t, y_test_t, features_t)

    return malaria_xgb, tb_xgb, features_m, features_t

with st.spinner("Loading prediction models..."):
    malaria_model, tb_model, features_m, features_t = load_models()

st.success("Models ready.")

tab1, tab2, tab3 = st.tabs(["Malaria Risk", "TB Treatment Failure", "Model Performance"])

with tab1:
    st.markdown("### Severe Malaria Risk Prediction")
    st.markdown("Enter patient details to predict risk of severe malaria outcome.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age (years)", 0, 85, 4)
        sex = st.selectbox("Sex", ["Male", "Female"])
        county = st.selectbox("County", [
            "Kisumu", "Homa Bay", "Migori", "Siaya",
            "Nairobi", "Mombasa", "Nakuru", "Kisii",
            "Turkana", "Kakamega", "Bungoma"
        ])
        iron_deficiency = st.checkbox("Iron deficiency present")
        bed_net_use = st.checkbox("Uses insecticide-treated bed net")

    with col2:
        hiv_positive = st.checkbox("HIV positive")
        prior_malaria = st.checkbox("Prior malaria episode")
        haemoglobin = st.slider("Haemoglobin (g/dL)", 4.0, 18.0, 10.5, 0.1)
        parasite_density = st.number_input("Parasite density (parasites/μL)",
                                            min_value=100, max_value=100000,
                                            value=5000, step=100)
        days_to_treatment = st.slider("Days from onset to treatment", 0, 14, 2)
        wealth_quintile = st.slider("Wealth quintile (1=poorest)", 1, 5, 2)

    if st.button("Predict Malaria Risk", type="primary"):
        high_risk_counties = ["Kisumu", "Homa Bay", "Migori", "Siaya"]
        input_data = pd.DataFrame([{
            "age": age,
            "iron_deficiency": int(iron_deficiency),
            "bed_net_use": int(bed_net_use),
            "hiv_positive": int(hiv_positive),
            "prior_malaria_episode": int(prior_malaria),
            "haemoglobin_gdl": haemoglobin,
            "parasite_density": parasite_density,
            "days_to_treatment": days_to_treatment,
            "wealth_quintile": wealth_quintile,
            "sex_male": int(sex == "Male"),
            "high_risk_county": int(county in high_risk_counties)
        }])

        prob = malaria_model.predict_proba(input_data)[0][1]
        risk_level = "HIGH" if prob > 0.6 else "MODERATE" if prob > 0.35 else "LOW"
        color = "red" if prob > 0.6 else "orange" if prob > 0.35 else "green"

        st.markdown(f"### Predicted Risk: :{color}[{risk_level}]")
        st.metric("Probability of Severe Malaria", f"{prob:.1%}")

        st.markdown("**Key risk factors for this patient:**")
        if parasite_density > 10000:
            st.warning("High parasite density — major risk factor")
        if days_to_treatment > 3:
            st.warning(f"Delayed treatment ({days_to_treatment} days) — increases severity risk")
        if not bed_net_use:
            st.info("No bed net use — consider provision")
        if county in high_risk_counties:
            st.warning(f"{county} is in a high malaria transmission zone")

with tab2:
    st.markdown("### TB Treatment Failure Risk Prediction")

    col1, col2 = st.columns(2)
    with col1:
        age_tb = st.slider("Age (years) ", 15, 80, 35)
        sex_tb = st.selectbox("Sex ", ["Male", "Female"])
        hiv_tb = st.checkbox("HIV positive ")
        mdr_tb = st.checkbox("MDR-TB confirmed")
        dot_therapy = st.checkbox("Receiving DOT (directly observed therapy)")

    with col2:
        diabetes_tb = st.checkbox("Diabetes")
        smoking_tb = st.checkbox("Current smoker")
        bmi_tb = st.slider("BMI", 13.0, 35.0, 19.5, 0.1)
        sputum_grade = st.selectbox("Sputum smear grade", [1, 2, 3])
        household_contacts = st.slider("Household contacts", 0, 10, 3)
        wealth_q_tb = st.slider("Wealth quintile (1=poorest) ", 1, 5, 2)
        urban_county = st.checkbox("Urban county (Nairobi, Mombasa, Nakuru)")

    if st.button("Predict TB Treatment Failure Risk", type="primary"):
        input_tb = pd.DataFrame([{
            "age": age_tb,
            "hiv_positive": int(hiv_tb),
            "mdr_tb": int(mdr_tb),
            "dot_therapy": int(dot_therapy),
            "diabetes": int(diabetes_tb),
            "smoking": int(smoking_tb),
            "bmi": bmi_tb,
            "sputum_grade": sputum_grade,
            "household_contacts": household_contacts,
            "wealth_quintile": wealth_q_tb,
            "sex_male": int(sex_tb == "Male"),
            "urban_county": int(urban_county)
        }])

        prob_tb = tb_model.predict_proba(input_tb)[0][1]
        risk_level_tb = "HIGH" if prob_tb > 0.5 else "MODERATE" if prob_tb > 0.25 else "LOW"
        color_tb = "red" if prob_tb > 0.5 else "orange" if prob_tb > 0.25 else "green"

        st.markdown(f"### Predicted Risk: :{color_tb}[{risk_level_tb}]")
        st.metric("Probability of Treatment Failure", f"{prob_tb:.1%}")

        if mdr_tb:
            st.error("MDR-TB confirmed — high priority for enhanced monitoring")
        if hiv_tb:
            st.warning("HIV co-infection — ensure ART is optimised")
        if not dot_therapy:
            st.warning("Not on DOT — directly observed therapy significantly improves outcomes")
        if bmi_tb < 18.5:
            st.warning(f"Low BMI ({bmi_tb}) — nutritional support recommended")

with tab3:
    st.markdown("### Model Performance and Fairness")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Malaria Model (XGBoost)**")
        st.metric("AUC-ROC", "0.686")
        st.metric("Training samples", "1,034 (SMOTE balanced)")
        st.markdown("**Fairness findings:**")
        st.markdown("- Female AUC: 0.739 vs Male AUC: 0.641")
        st.markdown("- Under 5 AUC: 0.906 (strongest subgroup)")
        st.markdown("- 25 to 44 age group: lowest performance (0.635)")

    with col2:
        st.markdown("**TB Model (XGBoost)**")
        st.metric("AUC-ROC", "0.642")
        st.metric("Training samples", "1,068 (SMOTE balanced)")
        st.markdown("**Fairness findings:**")
        st.markdown("- Male AUC: 0.681 vs Female AUC: 0.578")
        st.markdown("- 15 to 24 age group: 0 recall — model misses all young adult failures")
        st.markdown("- 55 to 64 age group: AUC 0.222 — near random performance")

    st.divider()
    st.markdown("**Responsible AI Note:**")
    st.info(
        "This tool is for research and educational purposes only. "
        "It should not be used for clinical decision-making without validation "
        "on real patient data. Performance gaps across demographic subgroups "
        "have been identified and must be addressed before any clinical deployment. "
        "All data used in training is synthetic, generated to reflect Kenya "
        "epidemiological distributions from KEMRI and DHIS2 sources."
    )

st.divider()
st.caption("ZENIK.AI · Kenya Malaria and TB Outcome Predictor · XGBoost + SHAP · Responsible AI")