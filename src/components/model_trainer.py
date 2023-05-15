import os
import sys
from dataclasses import dataclass

import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

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

            counts = np.unique(y_train, return_counts=True)[-1] # gets the count of the unique values in y_train
            scale_pos_weight = counts[0]/counts[1]
            # models = {
            #     "logreg" : LogisticRegression(class_weight="balanced"),
            #     "randomforest" : RandomForestClassifier(class_weight="balanced", random_state=1),
            #     "adaboost" : AdaBoostClassifier(random_state=1),
            #     "xgboost" : XGBClassifier(scale_pos_weight = scale_pos_weight, random_state=1),
            #     "svm" : SVC(class_weight="balanced", random_state=1),
            #     "tree" : DecisionTreeClassifier(random_state=1, class_weight="balanced")
            #     }
            
            # model_report:dict = evaluate_models(X_train, y_train,
            #                                    X_test, y_test, models=models)
            
            # best_model_name = sorted(
            #     model_report, key=lambda x: model_report[x]["auc_score"],
            #     reverse=True
            # )[1]
            
            # best_model_score = model_report[best_model_name]["auc_score"]
            # best_model = models[best_model_name]


            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")
            
            # logging.info("Found best model")

            logging.info("initialising best model")
            model = XGBClassifier(
                max_depth=3, min_child_weight=7, learning_rate=0.08,
                gamma=0.0, reg_alpha=0.1, subsample=0.75, scale_pos_weight=scale_pos_weight
            )
            
            model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=model
            )


            predictions = model.predict(X_test)
            auc_score = roc_auc_score(y_test, predictions)
            con_matrix = confusion_matrix(y_test, predictions)

            return auc_score, con_matrix

        except Exception as e:
            raise CustomException(e, sys)
        
        


