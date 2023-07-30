## Showcasing An End to End Machine Learning Project

## Goal
The goal of the project is to demonstrate a complete machine learning workflow starting from EDA,
to model training and then model deployment as well as writing tests.

## About The Data
The dataset used for this project was shared by <a href ="">Leonardo Martinelli</a> on kaggle.
The dataset is about school bullying and contains columns such as:

* `Bullied_on_school_property_in_past_12_months`
* `Bullied_not_on_school_property_in_past_12_months`
* `Cyber_bullied_in_past_12_months`
* `Custom_age`
* `Sex`
* `Physical_attack`
* `Physical_fighting`
* `Felt_lonely`

## Requirements
* pandas==2.0.0
* numpy==1.24.2
* seaborn==0.12.2
* matplotlib==3.7.1
* scikit-learn==1.2.2
* xgboost==1.7.5
* imbalanced-learn==0.10.1
* dill==0.3.6
* flask==2.3.2
* flask-wtf==1.1.1
* shap==0.41.0

## Model
The model trained on the data is a support vector machine classifier. 
Due to the imbalance nature of the data, the performance was evaluated on `roc_auc_score`, `recall_score` and not just on the accuracy.

## Web app
The web app was built using Flask framework in the backend and was hosted on https://www.render.com

You can visit the web app <a href="https://t.co/afn4AUcnvt">here</a>
