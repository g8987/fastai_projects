# Loan Default Risk Prediction

## Overview

This repository contains a loan default risk prediction project aimed at assisting investors on the Lending Club platform. By using machine learning algorithms, the model predicts the probability of borrowers defaulting on their loans, enabling investors to make more informed decisions.

## Project Statement

The main goal of this project is to develop a predictive model that can analyze borrower profiles and loan characteristics to assess the risk of loan default. The model's performance will be evaluated using various metrics to ensure its effectiveness. 

This machine learning model is intended to be used as a reference tool to help investors make informed decisions about lending to potential borrowers based on their ability to repay. The main purpose is to lower risk and maximize profit.

## Data Collection

The dataset used on this project contains more than 9,500 loans with information about the borrower profile, loan structure and whether the loan was repaid. This data was extracted from [Kaggle - Loan Data](https://www.kaggle.com/datasets/itssuru/loan-data). 


## Data dictionary

|    | Variable          | Explanation                                                                                                             |
|---:|:------------------|:------------------------------------------------------------------------------------------------------------------------|
|  0 | credit_policy     | 1 if the customer meets the credit underwriting criteria; 0 otherwise.                                                  |
|  1 | purpose           | The purpose of the loan.                                                                                                |
|  2 | int_rate          | The interest rate of the loan (more risky borrowers are assigned higher interest rates).                                |
|  3 | installment       | The monthly installments owed by the borrower if the loan is funded.                                                    |
|  4 | log_annual_inc    | The natural log of the self-reported annual income of the borrower.                                                     |
|  5 | dti               | The debt-to-income ratio of the borrower (amount of debt divided by annual income).                                     |
|  6 | fico              | The FICO credit score of the borrower.                                                                                  |
|  7 | days_with_cr_line | The number of days the borrower has had a credit line.                                                                  |
|  8 | revol_bal         | The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).                           |
|  9 | revol_util        | The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available). |
| 10 | inq_last_6mths    | The borrower's number of inquiries by creditors in the last 6 months.                                                   |
| 11 | delinq_2yrs       | The number of times the borrower had been 30+ days past due on a payment in the past 2 years.                           |
| 12 | pub_rec           | The borrower's number of derogatory public records.                                                                     |
| 13 | not_fully_paid    | 1 if the loan is not fully paid; 0 otherwise.   



## Machine Learning Models

Three different machine learning models were trained and compared: Logistic Regression, Random Forest, and XGBoost. Each model's performance was evaluated to determine the most suitable approach for loan default risk prediction.

## Model Evaluation and Performance

The models were evaluated using various evaluation metrics such as accuracy and F1-score. The model with the best performance was selected for further analysis.

## Hyperparameter Tuning

To optimize the selected model's performance, hyperparameter tuning was performed using techniques like Randomized Search Cross-Validation.


## Results and Insights

The final model achieved a decent F1-score in identifying potential loan defaulters, but there is still room for improvement. Insights were gained from feature importance analysis and partial dependence plots.
It's important to note that predicting loan defaults is a challenging task and achieving high accuracy for both classes simultaneously can be difficult.





