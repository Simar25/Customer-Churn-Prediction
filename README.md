# Customer Churn Prediction

Predicting telecom customer churn using classical ML models, with a focus on
handling class imbalance and explaining predictions with SHAP. Includes a
deployed Streamlit app for live predictions.

🔗 **Live Demo:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app)

## Problem Statement

Customer churn is one of the most expensive problems for subscription-based
businesses — acquiring a new customer typically costs far more than retaining
an existing one. This project builds a model to identify customers at high
risk of churning, so a business could proactively intervene (e.g., targeted
offers, outreach) before losing them.

## Dataset

[IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(Kaggle) — 7,043 customers, 21 features including demographics, account
information, and subscribed services. ~26% of customers in the dataset churned.

## Approach

1. **EDA** — explored churn rates across contract type, tenure, and monthly
   charges to identify early signals.
2. **Data Cleaning & Feature Engineering** — fixed data type issues,
   one-hot encoded categorical variables, engineered a `NumServices` feature
   (total add-on services subscribed).
3. **Class Imbalance Handling** — applied SMOTE on the training set only, to
   avoid biasing the model toward the majority (non-churn) class.
4. **Model Comparison** — trained and evaluated Logistic Regression, Random
   Forest, and XGBoost using precision, recall, F1, and ROC-AUC (not just
   accuracy, which is misleading on imbalanced data).
5. **Explainability** — used SHAP to interpret the best model's predictions,
   both globally (which features matter most) and locally (why a specific
   customer was flagged).
6. **Deployment** — packaged the final model into an interactive Streamlit
   app for live predictions.

## Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.xx | 0.xx | 0.xx | 0.xx |
| Random Forest | 0.xx | 0.xx | 0.xx | 0.xx |
| XGBoost | 0.xx | 0.xx | 0.xx | 0.xx |

*(Fill in with your actual numbers from the comparison table you generated in Phase 4.)*

**XGBoost** was selected as the final model based on [state your reason — e.g.
best recall on the churn class, since missing an actual churner is costlier
than a false alarm].

## Key Insights (from EDA + SHAP)

- Customers on **month-to-month contracts** churn far more than those on
  one- or two-year contracts.
- **Low tenure** (newer customers) is one of the strongest churn predictors —
  two-year contracts and long tenure are strongly protective.
- **Fiber optic internet** customers show elevated churn risk despite being a
  premium service — worth investigating pricing or service quality.
- Customers subscribed to **fewer add-on services** are more likely to churn,
  suggesting engagement with more services increases retention.

![SHAP Summary Plot](reports/shap_summary.png)

## Tech Stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, Streamlit

## How to Run Locally

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
