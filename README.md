# 📡 Telecom Customer Churn Prediction: Proactive Retention Pipeline



![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)

![XGBoost](https://img.shields.io/badge/XGBoost-Powerful-green.svg)

![License](https://img.shields.io/badge/License-MIT-yellow.svg)



## 🎯 Project Overview



In the telecom industry, retaining customers is significantly more cost-effective than acquiring new ones. This project develops a **professional-grade classification pipeline** to predict customer churn using the 'Telco-Customer-Churn' dataset (~7,000 records).



By leveraging advanced machine learning techniques, we transform raw customer data into actionable business intelligence, identifying high-risk individuals before they leave.



### 🚩 The Challenge: Imbalanced Data



A major hurdle in churn prediction is the **class imbalance**. In this dataset, only ~26% of customers actually churn. Standard models often fail to detect these "hidden" churners. We solved this by implementing **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the data and ensure our model learns the patterns of "Churners" just as well as "Loyalists."



## 🚀 Technical Victory: The Pipeline



Our modular architecture ensures reproducibility and scalability:



- **Data Engineering**: Handled messy data including median imputation for `TotalCharges` and intelligent label encoding for 16 categorical features.

- **Resampling Strategy**: Applied **SMOTE** to the training set, increasing the minority class visibility without duplicating data.

- **Model Suite**: Compared Logistic Regression, Random Forest, and **XGBoost**.

- **Performance**: Achieved a **ROC-AUC of 0.8328** and a high recall for churners, prioritizing the identification of at-risk customers.



## 📁 File Structure



```text

├── src/

│   ├── preprocessing.py    # Data cleaning, Imputation, Encoding

│   ├── models.py           # SMOTE logic & Model training (XGBoost, RF, LR)

│   └── visualization.py    # ROC Curves, Confusion Matrix, Feature Importance

├── main.py                 # Orchestrator script

├── requirements.txt        # Dependency management

└── Telecom_Churn_Analysis_Report.md # Full detailed project report

```



## 🛠️ How to Run



1. **Clone the repository**:

   ```bash

   git clone https://github.com/your-username/telecom-churn-prediction.git

   cd telecom-churn-prediction

   ```

2. **Install dependencies**:

   ```bash

   pip install -r requirements.txt

   ```

3. **Execute the pipeline**:

   ```bash

   python main.py

   ```



## 📊 Key Insights for Business Stakeholders



Through our **Feature Importance Analysis**, we discovered:



1.  **Contract Type**: Month-to-month contracts are the #1 predictor of churn.

2.  **Tenure**: Customers are most likely to leave within their first 6 months.

3.  **Monthly Charges**: High bills without service bundling (Tech Support/Security) lead to higher attrition.



### 💡 Proposed Retention Strategies:



- **Migration Incentives**: High-value discounts for customers moving from month-to-month to 1-year contracts.

- **Early-Tenure Concierge**: Proactive support calls in the first 90 days to increase product stickiness.

- **Safety Bundling**: Offering "Tech Support" as a loyalty perk rather than a surcharge.



---



**Developed with ❤️ for Data Science Excellence.**

_Ready for Internship Submission / Portfolio Showcase._

