# Predicting Bank Term Deposit Subscription with Machine Learning

## Project Overview

This project develops an end-to-end machine learning classification pipeline to predict whether a bank customer will subscribe to a term deposit before a marketing call is made.

The analysis uses the **Bank Marketing dataset from the UCI Machine Learning Repository**, containing 45,211 observations from direct marketing campaigns conducted by a Portuguese banking institution.

The project focuses on both predictive performance and realistic model deployment. In particular, the `duration` variable is excluded because it is only known after a marketing call occurs and would therefore introduce data leakage.

## Business Question

Can customer characteristics and historical marketing information be used to identify customers who are more likely to subscribe to a term deposit before a marketing call is made?

## Dataset

- **Observations:** 45,211
- **Original variables:** 17
- **Model predictors:** 15
- **Target:** Term deposit subscription (`yes` / `no`)
- **Positive class rate:** 11.70%
- **Data source:** UCI Machine Learning Repository — Bank Marketing Dataset

Because the target variable is highly imbalanced, model performance is evaluated using precision, recall, F1-score, and ROC-AUC rather than accuracy alone.

## Methodology

The machine learning workflow includes:

1. Data quality and duplicate checks
2. Data leakage identification and prevention
3. Train / validation / test split
4. Numerical feature scaling
5. One-hot encoding of categorical variables
6. Logistic Regression baseline model
7. Random Forest nonlinear model
8. Classification threshold optimization using the validation set
9. Final evaluation on an untouched test set
10. Feature importance and business insight analysis

The data were divided into:

| Dataset | Observations |
|---|---:|
| Training | 27,126 |
| Validation | 9,042 |
| Test | 9,043 |

## Model Comparison

Thresholds were selected on the validation set by optimizing F1-score.

| Model | ROC-AUC | Optimal Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7643 | 0.2061 | 0.4675 | 0.4282 | 0.4470 |
| Random Forest | **0.7813** | 0.2400 | 0.4382 | **0.4991** | **0.4666** |

Random Forest was selected as the final model based on validation F1-score.

## Final Test Performance

Using the selected classification threshold of **0.24**, the final Random Forest model achieved:

| Metric | Test Result |
|---|---:|
| ROC-AUC | **0.7932** |
| Precision | 0.4444 |
| Recall | 0.5180 |
| F1-score | **0.4784** |
| Accuracy | 0.8679 |

The model identified **548 of 1,058 actual subscribers** in the test set.

## Feature Importance

The five most influential original predictors in the Random Forest model were:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | Balance | 0.1684 |
| 2 | Age | 0.1425 |
| 3 | Day | 0.1271 |
| 4 | Month | 0.1100 |
| 5 | Job | 0.0921 |

## Key Business Insights

Exploratory analysis produced several notable patterns:

- Subscription rates increased from **7.24%** in the lowest balance quartile to **16.15%** in the highest.
- Customers aged **60+** had an observed subscription rate of **42.26%**.
- Customers with a successful previous campaign outcome had an observed subscription rate of **64.73%**.
- Customers contacted via cellular channels had a **14.92%** subscription rate, compared with **4.07%** for customers whose contact type was unknown.

These findings are descriptive associations and should not be interpreted as causal effects.

## Technical Highlights

- Built an end-to-end preprocessing and modeling pipeline using `scikit-learn`
- Prevented target leakage by excluding information unavailable at prediction time
- Used separate validation and test sets to avoid test-set contamination
- Addressed class imbalance through threshold optimization and appropriate evaluation metrics
- Compared an interpretable linear baseline with a nonlinear ensemble model
- Aggregated one-hot encoded feature importance back to original business variables

## Technologies

**Python** · **pandas** · **NumPy** · **scikit-learn** · **Matplotlib** · **Google Colab**

## Repository

The complete analysis, model development process, visualizations, and results are available in:

`Bank_Marketing_Subscription_Prediction.ipynb`

## Reproducibility

The notebook automatically downloads the Bank Marketing dataset from the UCI Machine Learning Repository and reproduces the complete analysis from raw data to final model evaluation.

A fixed random seed (`random_state=42`) is used where applicable to improve reproducibility.
