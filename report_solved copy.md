# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).
###########################################################################################################

The primary goal of this analysis was to develop machine learning models to predict loan risk using financial data.

The dataset includes financial attributes of loan applicants, such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt.

The aim is to classify loans into two categories: Healthy Loans (Class 0) and High-Risk Loans (Class 1), based on their risk factors or repayment potential.

The analysis started by collecting lending data from a CSV file, followed by splitting it into features and the target variable ('loan_status'). The data was then divided into training and testing sets for model training and evaluation.

For this task, the LogisticRegression method from the scikit-learn library was selected due to its effectiveness in binary classification. The model was trained on the training dataset, and predictions were made on the test data, with various metrics calculated to evaluate performance.

To address class imbalance, the RandomOverSampler technique from the imbalanced-learn library was applied. This approach generates synthetic samples to balance the dataset, particularly for the underrepresented high-risk loans, leading to improved model training and better accuracy in predicting high-risk loans.

###########################################################################################################

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.
###########################################################################################################

Machine Learning Model

Accuracy: 0.99
Balanced Accuracy Score: 0.952
Precision (Class 0 - Healthy Loan): 1.00
Recall (Class 0 - Healthy Loan): 0.99
Precision (Class 1 - High-Risk Loan): 0.84
Recall (Class 1 - High-Risk Loan): 0.94

              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.94      0.89       619

    accuracy                           0.99     19384
   macro avg       0.92      0.97      0.94     19384
weighted avg       0.99      0.99      0.99     19384


###########################################################################################################

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

###########################################################################################################
The Logistic Regression model performs well in predicting healthy loans. However, the test data reveals that it misclassifies 10% of high-risk loans as healthy. While the training data includes only loans with amounts under $24,000, even a single defaulted loan, especially early in its term, could result in losses greater than the interest earned from many healthy loans.

Another challenge is the limited number of high-risk loans in the dataset, which hinders the model's ability to learn effectively from this category. Increasing the number of high-risk loans in the training data could potentially enhance the model's performance.

When it comes to loan classification, prioritizing the prevention of high-risk loans is crucial. A single defaulted loan can result in greater financial losses than the interest gained from multiple healthy loans, making accurate identification of high-risk loans more important than simply issuing more loans.

If this model is only applied to loans under $24,000, and if the historical ratio of healthy to risky loans at the bank matches the 30-to-1 ratio observed in the data, then the model might be considered adequate. However, I would not recommend it for broader use due to the increased risk.