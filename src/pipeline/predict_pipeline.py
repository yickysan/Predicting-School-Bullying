import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data: pd.DataFrame) -> int:
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            new_data = preprocessor.transform(data)
            
            pred = model.predict(new_data)
            return pred
        
        except Exception as e:
            raise CustomException(e, sys)


class MakePredictDF:
    def  __init__(self, bullied_not_school: str, cyber_bullied: str,
                  custom_age: str, sex: str, close_friends: str, missed_school: str,
                  physically_attacked: str, physical_fighting: str, felt_lonely: str,
                  other_students_kind_and_helpful: str, parents_understand_problems: str
                  ):
        
        self.bullied_not_school = bullied_not_school
        self.cyber_bullied = cyber_bullied
        self.custom_age = custom_age
        self.sex = sex
        self.close_friends = close_friends
        self.missed_school =missed_school
        self.physically_attacked = physically_attacked
        self.physical_fighting = physical_fighting
        self.felt_lonely = felt_lonely
        self.other_students_kind_and_helpful = other_students_kind_and_helpful
        self.parents_understand_problems = parents_understand_problems

    def make_df(self) -> pd.DataFrame:
        try:
           data_dict =  {
                            "bullied_not_school" : [self.bullied_not_school],
                            "cyber_bullied" : [self.cyber_bullied],
                            "custom_age" : [self.custom_age],
                            "sex" : [self.sex],
                            "close_friends" : [self.close_friends],
                            "missed_school" : [self.missed_school],
                            "physically_attacked" : [self.physically_attacked],
                            "physical_fighting" : [self.physical_fighting],
                            "felt_lonely" : [self.felt_lonely],
                            "other_students_kind_and_helpful" : [self.other_students_kind_and_helpful],
                            "parents_understand_problems" : [self.parents_understand_problems]
                            }
           
           return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)

