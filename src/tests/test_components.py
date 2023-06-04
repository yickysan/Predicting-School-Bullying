import pytest
import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import clean_data, save_object, load_object
from src.exception import CustomException

@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    # Runs tests then cleans up the resulting file
    yield
    try:
        os.remove("src/tests/train.csv")
        os.remove("src/tests/test.csv")
        os.remove("src/tests/data.csv")
        os.remove("src/tests/obj.pkl")
        os.remove("src/tests/preprocessor.pkl")
        os.remove("src/tests/model.pkl")

    except Exception as e:
        raise CustomException(e, sys)


def test_load_and_save_object():
        obj_path = os.path.join("src", "tests", "obj.pkl")
        ohe = OneHotEncoder()

        try:
            save_object(obj_path, ohe)

        except Exception as e:
            raise CustomException(e, sys)
        
        try:
            load_object(obj_path)

        except Exception as e:
            raise CustomException(e, sys)




class TestComponents:
    """This class verifies that the data_ingestion, data_transformation and model trainer
      component works correctly
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def ingest():
        data_ingestion = DataIngestion()

        # specify new path to save the resulting data within the tests directory
        data_ingestion.ingestion_config.train_data_path = os.path.join("src", "tests", "train.csv")
        data_ingestion.ingestion_config.test_data_path = os.path.join("src", "tests", "test.csv")
        data_ingestion.ingestion_config.raw_data_path = os.path.join("src", "tests", "data.csv")

        data_ingestion.initiate_data_ingestion()

        return data_ingestion
    
    @staticmethod
    @pytest.fixture(scope="class")
    def read_dummy_data():
        return pd.read_csv("src/tests/test_dummy.csv")
    
    def test_clean_data(self, read_dummy_data):
        df = read_dummy_data
        clean_df = clean_data(df)
        for colname in ["were_obese", "were_underweight", "were_overweight"]:
            assert colname not in clean_df.columns

        null_sum = clean_df.isnull().sum().values.sum()
        assert null_sum == 0

    
    def test_data_ingestion(self, ingest):
        try:
            pd.read_csv(ingest.ingestion_config.train_data_path)
            pd.read_csv(ingest.ingestion_config.test_data_path)
            pd.read_csv(ingest.ingestion_config.raw_data_path)

        except Exception as e:
            raise CustomException(e, sys)


    @pytest.fixture(scope="class")
    def transform(self, ingest):
        train_path = ingest.ingestion_config.train_data_path
        test_path = ingest.ingestion_config.test_data_path

        data_transformation = DataTransformation()
        data_transformation.data_transformation_config.preprocessor_obj_file_path = os.path.join(
            "src", "tests", "preprocessor.pkl")
        
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

        return train_arr, test_arr

        

    def test_data_transformion(self, transform):
        train_arr, test_arr = transform
        
        assert type(train_arr) == type(test_arr)
        assert train_arr.shape[1] == test_arr.shape[1]

    
    @pytest.fixture(scope="class")
    def model_trainer(self, transform):
        train_arr, test_arr = transform

        model_trainer = ModelTrainer()
        model_trainer.model_trainer_config.model_file_path = os.path.join(
            "src", "tests", "model.pkl")
        
        auc_score, recall, _,_,_ = model_trainer.initiate_model_trainer(train_arr, test_arr)
        return auc_score, recall

    def test_model_performance(self, model_trainer):
        auc_score, recall = model_trainer
        assert auc_score >= 0.65
        assert recall >= 0.60

        

    






