# CUSTOMER_CHURN_PREDICTION

A machine learning project that predicts customer churn in the telecommunications industry using the IBM Telco Customer Churn dataset. Four classification models are trained, and evaluated, with Logistic Regression selected as the best-performing model based on AUC-ROC and Recall.

---

## Project Overview

Customer churn — when customers cancel or stop using a service. It is one of the most costly problems in the telecom industry. This project builds a binary classification pipeline to identify customers at risk of churning, enabling proactive retention strategies.

**Key result:** Logistic Regression achieved the highest AUC-ROC of **0.840** and a Recall of **0.802**, correctly identifying ~80% of customers who actually churned.

---

## Dataset

- **Source:** [WA_Fn-UseC_-Telco-Customer-Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM dataset via Kaggle
- **Records:** 7,043 customers
- **Features:** 21 columns (demographics, services, billing, contract type)
- **Target:** `Churn` (Yes / No) — imbalanced at 73.5% No, 26.5% Yes

---

## Methodology

**Preprocessing**
- Missing values in `TotalCharges` replaced with 0 (new customers with zero tenure)
- Categorical variables encoded using pandas `get_dummies` (One-Hot Encoding)
- Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) standardised with `StandardScaler` for KNN and Logistic Regression
- `class_weight='balanced'` applied to handle class imbalance in Logistic Regression, Decision Tree, and Random Forest

**Hyperparameter Tuning**
- `GridSearchCV` with 5-fold `KFold` cross-validation, scoring on F1
- KNN: best `k` selected by iterating k = 1–20 with 5-fold cross-validation

**Models Trained**
| Model | Tuning Method |
|---|---|
| K-Nearest Neighbours (KNN) | Cross-validated k selection (k=1–20) |
| Logistic Regression | Default hyperparameters + class weighting |
| Decision Tree | GridSearchCV (max_depth, min_samples_split, min_samples_leaf) |
| Random Forest | GridSearchCV (n_estimators, max_depth, min_samples_split, min_samples_leaf) |

---

## Results

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|---|---|
| KNN | 0.802 | 0.773 | 0.578 | 0.535 | 0.556 | 0.805 |
| Logistic Regression | 0.750 | **0.754** | 0.524 | **0.802** | **0.634** | **0.840** |
| Decision Tree | 0.765 | 0.746 | 0.515 | 0.743 | 0.608 | 0.821 |
| Random Forest | 0.855 | 0.769 | **0.551** | 0.711 | 0.621 | 0.832 |

**Best model: Logistic Regression** — highest AUC-ROC and Recall with minimal overfitting (train/test gap of 0.001).

---

## Key Findings

- Customers on **month-to-month contracts** churn at a significantly higher rate than those on longer-term plans
- **Short-tenure customers** (under 6 months) are the highest-risk group — early engagement is critical
- **Fiber optic subscribers** churn at roughly twice the rate of DSL users
- Higher **monthly charges** are associated with increased churn likelihood
- Customers paying via **electronic check** show higher churn rates than those using automated payment methods
- **Contract type, tenure, and TotalCharges** are the most important predictive features (Random Forest feature importance)

---

## Report

A full written report (`Customer_Churn_Prediction_Report.pdf`) is included, covering:
- Introduction and research objectives
- Dataset description
- Exploratory Data Analysis (EDA)
- Methodology and preprocessing steps
- Model evaluation and best model selection
- Feature importance analysis
- Conclusions and business recommendations

---

## Business Recommendations

1. **Contract incentives** — Offer discounts to migrate month-to-month customers to annual plans
2. **Early engagement** — Implement onboarding programmes targeting customers in their first 6 months
3. **Fibre optic review** — Investigate pricing and service quality for Fiber optic subscribers
4. **Payment method nudges** — Encourage customers to switch from electronic check to automated payment methods
5. **CRM integration** — Deploy the trained model to generate weekly churn risk scores for the retention team

---

## References

- IBM. (n.d.). *Telco Customer Churn* [Dataset]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Verbeke, W., et al. (2012). New insights into churn prediction in the telecommunication sector. *European Journal of Operational Research*, 218(1), 211–229.

