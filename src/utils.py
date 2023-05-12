import os
import sys
import numpy as np
import pandas as pd
import dill
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
    

def preprocessor(data, cat_columns, num_columns):
    """
    Function to perform preprocessing on the dataset.
    """
    try:
        # convert numerical columns into integer
        for col in num_columns:
            data[col] = data[col].str.extract("(\d+)").astype(int)


        # convert categorical columns into category data type
        for col in cat_columns:
            data[col] = data[col].astype("category").cat.codes

        return data.values
    
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