import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.loggers import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_file_obj_path=os.path.join('artifacts','preprocessor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns=[
                'Age',
                'Salt_Intake',
                'Stress_Score',
                'Sleep_Duration',
                'BMI'
                ]
            categorical_columns=[
                'BP_History',
                'Family_History',
                'Exercise_Level',
                'Smoking_Status',
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical preprocessing pipeline done")
            logging.info("Categorical preprocessing pipeline done")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            label_enc=LabelEncoder()
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            preprocessor_obj=self.get_data_transformer_object()

            logging.info("Train and Test dataset obtained")
            logging.info("preprocessor object created")

            target_column_name="Has_Hypertension"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=label_enc.fit_transform(train_df[target_column_name])

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=label_enc.fit_transform(test_df[target_column_name])

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Preprocessing completed over the train and test datasets")

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_obj_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_obj_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        