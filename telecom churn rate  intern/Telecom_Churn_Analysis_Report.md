# Project Report: Telco Customer Churn Prediction Analysis

**Prepared for:** Internship Submission / Portfolio Excellence
**Role:** Senior Data Scientist
**Date:** February 11, 2026

---

## 1. Executive Summary

This report presents a comprehensive end-to-end Machine Learning solution for predicting customer churn for a telecommunications provider. Utilizing the 'Telco-Customer-Churn' dataset of approximately 7,043 records, we developed a sophisticated classification pipeline. The core objective was to identify at-risk customers with high precision, enabling proactive retention strategies.

Our methodology involved rigorous data preprocessing, including median imputation for missing values and handling class imbalance (initial 26.5% churn rate) using Synthetic Minority Over-sampling Technique (SMOTE). We evaluated multiple algorithms—Logistic Regression, Random Forest, and XGBoost. The **XGBoost** model emerged as the superior solution, achieving a **ROC-AUC of 0.8328**, signaling strong discriminative power. High-impact churn drivers identified include **Contract Type**, **Tenure**, and **Payment Method**.

---

## 2. Introduction

Customer churn, the rate at which customers stop doing business with an entity, is a critical KPI in the highly competitive telecom sector. Acquiring a new customer is often 5-25 times more expensive than retaining an existing one. This project leverages predictive analytics to shift the paradigm from reactive support to proactive retention.

### 2.1 Problem Statement

The dataset presents a typical imbalanced classification problem. Majority of the customers (approx. 74%) do not churn, which can lead standard models to be biased towards the majority class, overlooking the critical "Churn = Yes" signals.

---

## 3. Methodology

### 3.1 Data Acquisition & Inspection

The dataset contains 7,043 customer records with 21 features, including demographic info (gender, senior citizen status), account info (tenure, contract, payment method), and service info (Internet service, Technical support, etc.).

### 3.2 Data Cleaning & Preprocessing

- **Missing Value Management**: The `TotalCharges` column contained hidden null values represented as blank spaces. These were converted to numeric format, and a **Median Imputation** strategy was applied to maintain data integrity without introducing significant variance.
- **Label Encoding**: Categorical features were transformed into numerical representations using Label Encoding, ensuring compatibility with mathematical modeling while preserving the ordinal nature of certain variables where applicable.
- **Feature Selection**: High-cardinality identifiers like `customerID` were removed to prevent data leakage and noise.

### 3.3 Addressing Class Imbalance: SMOTE

To tackle the 20-26% churn imbalance, we implemented **SMOTE (Synthetic Minority Over-sampling Technique)**. Instead of simple oversampling (duplicating records), SMOTE creates synthetic instances of the minority class by interpolating between neighboring samples. This expanded our training set, allowing the models to learn the specific characteristics of churners more effectively.

### 3.4 Data Splitting

A **80/20 Stratified Split** was utilized. Stratification ensures that both training and testing sets maintain the same proportion of churners as the original dataset, providing a realistic evaluation environment.

---

## 4. Model Development & Evaluation

We compared three paradigms of classification:

1.  **Logistic Regression**: Serving as our baseline model.
2.  **Random Forest**: An ensemble method to capture non-linear relationships.
3.  **XGBoost (Extreme Gradient Boosting)**: Optimized for speed and performance through gradient boosting.

### 4.1 Evaluation Metrics

- **Confusion Matrix**: Crucial for understanding the trade-off between False Positives (over-spending on retention) and False Negatives (missing a customer who actually churns).
- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**: Our primary metric. An AUC of 0.8328 indicates that there is an 83.28% probability that the model will correctly rank a random churner higher than a random non-churner.

---

## 5. Key Findings & Discussion

### 5.1 Analysis of Churn Drivers

The feature importance plots revealed consistent patterns:

- **Contract (Month-to-month)**: This was the highest predictor of churn. Customers without long-term commitments are significantly more volatile.
- **Tenure**: Customers in the early stages of their lifecycle (0-6 months) exhibit the highest churn probability.
- **Total Charges / Monthly Charges**: Higher price points correlate with increased churn, suggesting a sensitivity to service value.

---

## 6. Strategic Business Recommendations

Based on the data-driven insights, we suggest three pillars of retention:

### 6.1 "Frictionless" Contract Migration

**Insight**: Month-to-month customers are "Flight Risks."
**Strategy**: Offer targeted "Value Upgrades" specifically for users on month-to-month plans at their 6-month mark. Provide a one-time credit (e.g., $50) for switching to a 12-month commitment.

### 6.2 Early Lifecycle Intervention

**Insight**: The first 6 months are critical for "Stickiness."
**Strategy**: Implement a "New User Concierge" program. Proactive technical support calls at month 1 and automated "Loyalty Perk" notifications at month 4.

### 6.3 Strategic Service Bundling

**Insight**: Customers with "Tech Support" and "Online Security" churn less.
**Strategy**: Cross-sell these services as a "Safety Bundle." For high-risk customers, offer these add-ons at a 50% discount for 6 months rather than a direct bill reduction. This increases the cost of switching for the customer while maintaining the core revenue.

---

## 7. Conclusion

The developed pipeline successfully achieves a professional standard for churn prediction. With a ROC-AUC of 0.8328, the model provides a reliable foundation for the marketing and customer success teams to deploy targeted interventions. Future iterations could involve deep learning or hyperparameter tuning via Bayesian Optimization to push the AUC beyond 0.85.

---

**End of Report**
