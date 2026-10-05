# Kenya Malaria & TB Outcome Predictor

A clinical prediction tool for severe malaria and TB treatment failure in Kenya. Built using XGBoost with SHAP explainability and a demographic fairness audit across sex and age subgroups.

## Live Demo
[your Streamlit URL here — update after deployment]

## What it does
- Predicts probability of severe malaria outcome based on 11 clinical and demographic features
- Predicts TB treatment failure risk based on 12 features including HIV status, MDR-TB, and DOT therapy
- Shows key risk factors driving each individual prediction
- Includes model performance and fairness audit across demographic subgroups

## Tech Stack
XGBoost · LightGBM · Scikit-learn · SHAP · imbalanced-learn · Streamlit · Python 3.11

## Data
Synthetic patient data generated to reflect Kenya epidemiological distributions from KEMRI and DHIS2. 1,000 malaria patients across 16 counties, 800 TB patients.

## Responsible AI
This tool is for research and educational purposes only. Performance gaps across demographic subgroups have been identified — the model performs significantly worse for females with TB (AUC 0.578) and for young adults aged 15 to 24 (0 recall for TB treatment failure). These gaps must be addressed before any clinical deployment.

## Portfolio Context
Project 3 of a Computational Epidemiology and Health Data Science portfolio under ZENIK.AI — building AI tools for African health research and clinical contexts.

GitHub: github.com/Mwauramos | Contact: mwauramos.n@gmail.com