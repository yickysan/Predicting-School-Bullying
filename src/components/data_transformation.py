import sys
import os
from dataclasses import dataclass 
import pandas as pd
import numpy as np

from src.utils import preprocessor, save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            
            cat_columns = ["bullied_not_school", "cyber_bullied", "sex",
                            "physically_attacked", "physical_fighting", "felt_lonely",
                            "other_students_kind_and_helpful", "parents_understand_problems"
                            ]
            
            num_columns = ["custom_age", "close_friends", "missed_school"]
            
            logging.info(f"Categorical columns: {cat_columns}")
            logging.info(f"Numerical Columns: {num_columns}")

            preprocessor_fn = preprocessor

            return (preprocessor_fn,
                    cat_columns,
                    num_columns
            )
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test dataset completed.")

            logging.info("Initiating preprocessor object.")
            
            preprocessing_obj = self.get_data_transformer_object()
            preprocessing_fn, cat_cols, num_cols = preprocessing_obj # unpack preprocesing_obj
            

            target = "bullied_in_school"
            train_feature_df = train_df.drop(target, axis=1)
            train_target = train_df[target].astype("category").cat.codes

            test_feature_df = test_df.drop(target, axis=1)
            test_target = test_df[target].astype("category").cat.codes

            logging.info("Applying preprocessing on the train and test dataset")

            train_feature_arr = preprocessing_fn(data=train_feature_df,
                                                 cat_columns=cat_cols,
                                                 num_columns=num_cols )
            test_feature_arr = preprocessing_fn(data=test_feature_df,
                                                cat_columns=cat_cols,
                                                num_columns=num_cols)

            train_arr = np.c_[
                train_feature_arr, np.array(train_target)
                ]
            

            test_arr = np.c_[
                test_feature_arr, np.array(test_target)
                ]
            
            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
            

        except Exception as e:
            raise CustomException(e, sys)

