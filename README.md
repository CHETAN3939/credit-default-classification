# Credit Card Default Classification using Machine Learning

---

## 1. Problem Statement

The objective of this project is to predict whether a credit card client will default on payment in the next month using various machine learning classification models. The goal is to compare different models and evaluate their performance using multiple evaluation metrics.

---

## 2. Dataset Description

Dataset Used: **Default of Credit Card Clients Dataset (UCI Repository)**

- Number of Instances: 30,000
- Number of Features: 23
- Target Variable: `default payment next month`
  - 0 → No Default
  - 1 → Default

The dataset contains demographic details, credit limits, bill amounts, payment history, and repayment records of credit card clients.

This is a binary classification problem.

---

## 3. Models Implemented

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble – Bagging)  
6. XGBoost (Ensemble – Boosting)  

---

## 4. Model Comparison Table

| ML Model Name        | Accuracy | AUC     | Precision | Recall  | F1 Score | MCC     |
|----------------------|----------|---------|-----------|---------|----------|---------|
| Logistic Regression  | 0.8077   | 0.7076  | 0.6868    | 0.2396  | 0.3553   | 0.3244  |
| Decision Tree        | 0.7157   | 0.6021  | 0.3676    | 0.3964  | 0.3814   | 0.1974  |
| KNN                  | 0.7928   | 0.7014  | 0.5487    | 0.3564  | 0.4322   | 0.3233  |
| Naive Bayes          | 0.4160   | 0.6516  | 0.2496    | 0.8176  | 0.3824   | 0.1111  |
| Random Forest        | 0.8095   | 0.7556  | 0.6211    | 0.3557  | 0.4523   | 0.3669  |
| XGBoost              | 0.8118   | 0.7565  | 0.6289    | 0.3640  | 0.4611   | 0.3764  |

---

## 5. Model Performance Observations

| ML Model Name | Observation |
|---------------|------------|
| Logistic Regression | Performs well as a baseline linear model with good precision, but recall is relatively low, meaning it misses some default cases. |
| Decision Tree | Captures non-linear patterns but shows lower overall performance and moderate overfitting tendencies. |
| KNN | Provides balanced performance but is sensitive to scaling and computationally heavier for large datasets. |
| Naive Bayes | Achieves very high recall but very low precision and accuracy, indicating it predicts many defaults incorrectly. |
| Random Forest | Improves performance significantly compared to Decision Tree by reducing overfitting through bagging. Shows strong AUC and MCC values. |
| XGBoost | Achieves the best overall performance with the highest Accuracy, AUC, F1 Score, and MCC, making it the most effective model for this dataset. |

Overall, ensemble methods (Random Forest and XGBoost) outperform individual models, with XGBoost showing the best predictive capability.

---

## 6. Streamlit Web Application Features

The deployed Streamlit application includes:

- Dataset upload option (CSV test data)
- Model selection dropdown
- Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix
- Classification report

The application is deployed using Streamlit Community Cloud.

---

## 7. Technologies Used

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Streamlit
- Joblib

---

## 8. Deployment

The application is deployed on Streamlit Community Cloud and connected to the GitHub repository containing the complete source code and saved models.


Deployed using Streamlit Community Cloud.




