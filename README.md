# Credit-card-fraud-detection
Credit card fraud detection systems are designed to identify and prevent fraudulent transactions made using credit or debit cards. These systems use various techniques such as data mining, machine learning, and artificial intelligence to detect patterns of fraudulent behavior and identify potentially fraudulent transactions in real-time.

# Content
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.


An open-source project for based on Fraud Detection using ML

![image](https://user-images.githubusercontent.com/103196507/197158948-e4090a6f-f69a-4e44-9b0d-c79157601bfc.png)


![image](https://user-images.githubusercontent.com/103196507/197159084-1f49af21-2edb-487b-9b35-e12ecf251913.png)


![image](https://user-images.githubusercontent.com/103196507/197159164-1fadb05a-9c0e-45da-a437-f29e1fcc5159.png)


![image](https://user-images.githubusercontent.com/103196507/197159235-4633355a-ef66-4663-9ca5-788fe74e663e.png)


![image](https://user-images.githubusercontent.com/103196507/197159301-508f237e-9b10-4418-acf2-11c7d1b25843.png)


![image](https://user-images.githubusercontent.com/103196507/197159388-e912f76f-e937-4142-9cd0-b9f71b69179d.png)


## Workflow

![image](https://user-images.githubusercontent.com/103196507/197159648-93fa4b19-4940-465b-9c9a-24e434c46305.png)



## Tech stack 
- Python 3

<p align="#">
<img alt="Python" src="https://img.shields.io/badge/python3-%23e4626b.svg?style=for-the-badge&logo=python3&logoColor=%23F7DF1E"/>
  </p>



