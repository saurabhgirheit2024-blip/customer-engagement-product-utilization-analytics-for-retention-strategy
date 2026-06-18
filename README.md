# Customer Engagement & Product Utilization Analytics: Churn Intelligence Portal
  
**Organization:** Unified Mentor Pvt Ltd / The European Central Bank  
**Dataset:** European Bank Customer Churn (`data/Churn_Modelling.csv`)  

---

## 1. Executive Summary

This enterprise-grade application analyzes bank customer churn and engagement dynamics across 10,000 customers. By integrating predictive machine learning models, unsupervised K-Means clustering, and a composite relationship health index, it transitions standard historical analysis into active **Decision Support** for bank managers and senior stakeholders.

### Core Strategic Findings:
1. **The Inactivity Hazard:** Inactive members exhibit a **1.89x** higher churn rate compared to active members. Lack of active transaction logs or mobile touchpoints is the primary operational metric linked to churn.
2. **The Portfolio U-Curve:** Customer churn rate follows a classic U-shaped curve based on product holdings. Holding exactly **2 products is the retention sweet spot** (associated with a churn rate of only **7.6%**). Single-product clients carry high churn (27.7%), while clients with 3 or 4 products spike to near-total loss (82.7% - 100%), representing operational friction or fee-related exit triggers.
3. **At-Risk Premium Segment:** Inactive high-balance accounts (Balances > $100,000) show a critical **31.3%** churn rate, representing immediate capital flight dangers.
4. **Predictive Reliability:** Our Support Vector Machine (SVM) and Random Forest models, trained on a stratified **80/20 train-test split** and balanced via **SMOTE**, identify customer churn with high sensitivity (71.3% and 65.1%) and Area Under the ROC Curve (AUC) of 0.838 - 0.841, providing highly credible decision vectors.

---

## 2. Research Paper & Analytical Methodology

### Data Ingestion & Integrity Checks
The data pipeline in `data_loader.py` enforces absolute mathematical and statistical validity prior to model fitting:
* **Demographic Boundaries:** Customers must be $\ge 18$ and $\le 120$ years of age.
* **Financial Boundaries:** Bank balances and annual estimated salaries must be non-negative ($\ge 0$).
* **Binary Consistency:** Binary variables (`HasCrCard`, `IsActiveMember`, `Exited`) must contain strictly `0` or `1`.
* **Completeness:** Asserts that zero null values exist across core analytical dimensions.

### Feature Engineering
To capture complex non-linear behavioral relationships, we introduce three engineered variables:
* **TenureByAge:** The ratio of tenure to the customer's age, indicating relative relationship duration.
* **BalanceSalaryRatio:** The ratio of account balance to estimated annual salary, highlighting wealth concentration.
* **CreditScoreGivenAge:** The ratio of credit score to age, proxying creditworthiness normalized over life stage.

### Class Imbalance & Resampling (SMOTE)
With only 20.37% of customers churned, the target variable is highly imbalanced. We implement **Synthetic Minority Over-sampling Technique (SMOTE)** to prevent classifier bias, significantly boosting the model's sensitivity (recall) in detecting true churners.

### Machine Learning Classification Models (80/20 Split)
We partition the dataset into an **80% training set (8,000 profiles)** and a **20% test set (2,000 profiles)**, evaluating four models under standard and resampled contexts:
1. **Logistic Regression:** Standard baseline for odds ratio evaluation.
2. **Support Vector Machine (SVM):** Non-linear kernel classifier maximizing margins on scaled features.
3. **Random Forest Classifier:** Ensemble of 100 decision trees (max depth 8) for capturing attribute interactions.
4. **eXtreme Gradient Boosting (XGBoost):** Gradient-boosted sequential decision trees for state-of-the-art accuracy.

---

## 3. Streamlit Dashboard Architecture

The dashboard frontend is organized into 3 primary interactive tabs:
1. **📊 Data Analysis & Cohorts**: Categorical/numerical exploratory charts, unsupervised K-Means segments ($K=4$), and interactive cohort heatmaps.
2. **🧪 Model Analysis & Diagnostics**: Performance comparison matrices (with and without SMOTE), interactive ROC curves, confusion matrices, and ranked feature drivers.
3. **🎯 Prediction & Simulator**: Live individual risk scorecard (displays predictions for all 4 models and the Relationship Strength Index), automated recommendations engine, strategic retention simulator (projected capital saved), and secure analyst export tools.

### Analyst Portal Access:
* **Username:** `Saurabh`
* **Password:** `123456`

---

## 4. Operational Launch & Execution Guide

### Prerequisites
Make sure all dependencies are pre-installed in your environment using the provided manifest:
```bash
pip install streamlit pandas numpy altair scikit-learn xgboost imbalanced-learn
```

### Relaunching the Server
To launch the interactive dashboard on your local Windows system, execute the following cmdlet in your PowerShell terminal:
```powershell
python -m streamlit run app.py
```
Streamlit will initialize, load and validate the custom data from `data/Churn_Modelling.csv`, train all ML models, run K-Means clustering, and open the active web application at:
👉 **https://m6xt7iwx5n9d4suwhutsg8.streamlit.app/**
