## Overview of the Analysis

# Purpose of the Analysis

The purpose of this analysis is to build and evaluate a logistic regression model to predict the likelihood of loan default based on various financial features. 
This predictive model aims to help a lending company identify high-risk loans and mitigate financial risks.

# Financial Information and Prediction Goal

The dataset contains financial information about loans, including borrower characteristics and loan attributes. 
The primary goal is to predict the 'loan_status', where a value of '0' indicates a healthy loan and a value of '1' indicates a high-risk loan.

# Basic Information about the Variables

*  loan_status: Target variable indicating loan status ('0' = healthy, '1' = high-risk).

*  Features: Various financial attributes of loans and borrowers, such as loan size, interest rate, borrowers income, etc.


# Stages of the Machine Learning Process


## 1.Data Loading and Preparation:


* Read the data from the Lending_data.csv file which is stored in Resources folder

* Separate the target variable (y) from the features (X).

* Split the data into training and testing sets using train_test_split.


## 2.Model Selection and Training:

* Use the LogisticRegression algorithm to train a model on the training data (X_train and y_train).

## 3.Model Prediction and Evaluation:

* Make predictions on the testing data (X_test).

* Evaluate the model's performance using metrics like confusion matrix, classification report, and accuracy score.

# Methods Used

* Logistic Regression: A statistical method used for binary classification to predict the probability of a binary outcome based on one or more predictor variables.

# Results

## Machine Learning Model 1: Logistic Regression

* Accuracy Score: 0.993

* Precision:

   *  Healthy Loan (0): 1.00
   
   *  High-Risk Loan (1): 0.86
* Recall:

   *  Healthy Loan (0): 0.99
   
   *  High-Risk Loan (1): 0.94
   
* F1-Score:

  * Healthy Loan (0): 1.00
  
  * High-Risk Loan (1): 0.90
  
# Summary

# Model Performance

The logistic regression model performs exceptionally well, achieving an overall accuracy of 99.3%. The model shows:

* Perfect precision (1.00) and high recall (0.99) for predicting healthy loans (0).

* High precision (0.86) and recall (0.94) for predicting high-risk loans (1).

# Recommendation

Based on the performance metrics, I recommend using the logistic regression model for predicting loan defaults. The model's high accuracy and strong precision and recall scores for both healthy and high-risk loans indicate its effectiveness and reliability. This model can be beneficial for the lending company to identify high-risk loans accurately and take appropriate measures to mitigate financial risks.

# Importance of Predicting Each Class

The model's ability to accurately predict both healthy and high-risk loans is crucial. While predicting high-risk loans (1) is essential to prevent defaults and mitigate risks, accurately identifying healthy loans (0) is equally important to ensure that potential customers are not wrongly flagged as high-risk, which could affect the company's lending operations and customer satisfaction.


## Technologies Used

- Python

- Pandas

- NumPy

- scikit-learn

- Jupyter Notebook




## Folder Structure 

The credit-risk classification Challenge consists of the following folders and files:

* **Credit_Risk folder**:

      * **Resources folder** :

           * **lending_data.csv** - The dataset used for training and testing the model.

      * **credit_risk_classfication.jpynb**   - The Jupyter Notebook file containing the code & analysis.
 

* **README.md**- Provides the Overview of the Analysis and project folder Structure


## How to Use

1. Clone the GitHub repository to your local machine using the following command:

git clone https://github.com/SriPenumatcha/credit-risk-classification.git

2. Open the `credit_risk_classfication.jpynb' file using Jupyter Notebook.

3. Run each cell in the notebook to perform the analysis and view the  results.

4. Review the analysis findings and conclusions in the notebook and the README.md file

