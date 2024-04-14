# Credit-Card-Fraud-Detection

This project aims to develop machine learning algorithms that can detect fraudulent credit card transactions accurately. The ultimate goal is to help credit card companies and customers avoid financial loss by identifying fraudulent transactions.

## Table of Contents
Introduction
Libraries Used
Data Preprocessing
Data Exploration
Model Training and Evaluation
Usage
Contributing

## Introduction
The dataset used for this project contains over 550,000 records of credit card transactions made by European cardholders in 2023. To protect the cardholders' identities, the data has been anonymized. The main purpose of this dataset is to assist in the development of fraud detection algorithms and models for identifying potentially fraudulent transactions.

The dataset is taken from kaggle [here](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data).

## Libraries Used
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn

## Data Preprocessing
By utilizing the power of Pandas, we are able to effortlessly load the dataset from a CSV file. We then proceed to meticulously clean the data by checking for any missing values and removing any rows containing such data. Moreover, we delve deep into the dataset to gain a profound understanding of its structure and distribution, thereby enabling us to extract valuable insights that can help drive progress and innovation.

## Data Exploration
In this section, we will conduct a thorough analysis of the data to gain a comprehensive understanding of its characteristics and distributions. We will also create a visual representation of the class distribution to evaluate any potential imbalance in the data. Additionally, we will plot the density of transactions over time for both fraudulent and non-fraudulent transactions. This will enable us to identify any discernible patterns or anomalies in the transaction data.

## Model Training and Evaluation
To detect credit card fraud, we employ various machine learning algorithms. We train and evaluate the following models:

Logistic Regression Random Forest Classifier K-mean clustering

We use evaluation metrics like ROC AUC score to assess the performance of each model and select the best one for fraud detection.

## Usage
To use this project, follow these steps:

1.Clone the repository.
2.Install the required libraries mentioned above.
3.Download the credit card fraud dataset.
4.Run the Jupyter Notebook or Python script to preprocess the data, train the models, and evaluate their performance.
5.Adjust the hyperparameters, algorithms, or feature engineering techniques based on your requirements.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.
