import os
import sys
from dataclasses import dataclass

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, accuracy_score, precision_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting the training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1], # takes every row and column except for the last column
                train_arr[:,-1], # takes every row in the last column
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # From the model evaluation done on the data set on the model_training notebook,
            # The logistic regression model and the xgboost model were the two best models.
            # After hyperparemeter tuning, the xgboost model was the best of the two.
            # The xgboost model with the required parameters will be used to fit the data

            logging.info("initialising best model")

            counts = np.unique(y_train, return_counts=True)[-1]
            scale_pos_weight = counts[0]/counts[1]

            model = XGBClassifier(
                max_depth=5, min_child_weight=1, learning_rate=0.08,
                gamma=0.0, reg_alpha=0.1, subsample=0.75, scale_pos_weight=scale_pos_weight
            )
            
            model.fit(X_train, y_train)

            logging.info("saving best model")

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=model
            )


            predictions = model.predict(X_test)
            auc_score = roc_auc_score(y_test, predictions)
            con_matrix = confusion_matrix(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            accuracy = accuracy_score(y_test, predictions)

            return(
                auc_score,
                recall,
                precision,
                accuracy,
                con_matrix
            )

        except Exception as e:
            raise CustomException(e, sys)
        
        


