import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import roc_auc_score, confusion_matrix

from src.exception import CustomException

def clean_data(data):
    """
    Function to clean the dataset
    """
    try:
        data.drop(
             ["were_obese", "were_underweight", "were_overweight"],
             axis=1, inplace=True
             )
        data.dropna(inplace=True)

        return data
    
    except Exception as e:
        raise CustomException(e, sys)
    
       
def save_object(file_path, obj):
    """
    Function to create and save pickle files.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for k, v in models.items():
            scores = {}
            model_name = k
            model = v
            model.fit(X_train, y_train) # fit the model with training data
            pred = model.predict(X_test)
            scores["auc_score"] = roc_auc_score(y_test, pred)
            scores["confusion_matrix"] = confusion_matrix(y_test, pred)
            report[model_name] = scores

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
        
    except Exception as e:
        raise CustomException(e, sys)
        