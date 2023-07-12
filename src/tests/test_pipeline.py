import pytest
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, MakePredictDF
from src.exception import CustomException

class TestPipeline:
    """This class tests that the predict pipeline works correctly"""


    @staticmethod
    @pytest.fixture(scope="class")
    def create_df() -> pd.DataFrame:
        make_df = MakePredictDF(
            bullied_not_school = "Yes",
            cyber_bullied= "Yes",
            custom_age= "15 years old",
            sex = "Female",
            close_friends="2",
            missed_school= "3 to 5 days",
            physically_attacked= "0 times",
            physical_fighting= "2 or 3 times",
            felt_lonely= "Rarely",
            other_students_kind_and_helpful= "Sometimes",
            parents_understand_problems= "Sometimes"
        )

        df = make_df.make_df()
        return df

    def test_make_df(self, create_df: callable):
        assert type(create_df) == type(pd.DataFrame())


    def test_predict_pipeline(self, create_df: callable):
        predict_pipeline = PredictPipeline()
        
        try:
            predict_pipeline.predict(create_df)
        
        except Exception as e:
            raise CustomException(e, sys)
