import pickle
import shutil
from sklearn import preprocessing
from forest_cover.exception import forest_cover_exception
from forest_cover.logger import logging
from forest_cover.entity.config_entity import DataTransformationConfig 
from forest_cover.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
import numpy as np
from forest_cover.constant import *
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,KNNImputer
import pandas as pd
from forest_cover.constant import *
from forest_cover.util.util import *



class DataTransformation:
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise forest_cover_exception(e,sys) from e
    def get_transformer_object_and_test_train_dir(self):
        try:
            os.makedirs(self.data_transformation_config.transformed_train_dir,exist_ok=True)
            logging.info(f"Train directory created-->{self.data_transformation_config.transformed_train_dir}")
            os.makedirs(self.data_transformation_config.transformed_test_dir,exist_ok=True)
            logging.info(f"Test directory created-->{self.data_transformation_config.transformed_test_dir}")
            os.makedirs(self.data_transformation_config.preprocessed_object_folder_path,exist_ok=True)
            logging.info(f"Preprocessing folder path created {self.data_transformation_config.preprocessed_object_folder_path}")
            #scaling on train data
            df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Reading the train file path which is {df}")
            df.drop(["Id","Soil_Type8","Soil_Type7",'Soil_Type1','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type2','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Soil_Type5','Soil_Type6','Soil_Type9','Wilderness_Area4','Hillshade_3pm'],axis=1,inplace=True)
            logging.info("Dropping columns 'Id','Soil_Type8','Soil_Type7','Soil_Type1','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type2','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Soil_Type5','Soil_Type6','Soil_Type9','Wilderness_Area4','Hillshade_3pm'")
            X=df.drop("Cover_Type",axis=1)
            y=df["Cover_Type"]
            logging.info("Separation as X and Y")
            num_pipeline = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=2)),('scaler', StandardScaler())])
            logging.info("Pipeline created!!")
            numerical_columns=NUMERICAL_COLUMNS
            logging.info("Standard scaler function called")
            preprocessing = ColumnTransformer([('num_pipeline',num_pipeline, numerical_columns)])
            logging.info(f"{X.columns}")
            l=list(X.columns)
            p=list(numerical_columns)
            df=pd.concat([pd.DataFrame(preprocessing.fit_transform(X),columns=X.columns),y],axis=1)
            df.to_csv(os.path.join(self.data_transformation_config.transformed_train_dir,"train.csv"),index=False)
            #scaling on test data
            df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Reading the test file path which is {df}")
            df.drop(["Id","Soil_Type8","Soil_Type7",'Soil_Type1','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type2','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Soil_Type5','Soil_Type6','Soil_Type9','Wilderness_Area4','Hillshade_3pm'],axis=1,inplace=True)
            logging.info("Dropping columns 'Id','Soil_Type8','Soil_Type7' ")
            X=df.drop("Cover_Type",axis=1)
            y=df["Cover_Type"]
            logging.info("Separation as X and Y")
            df=pd.concat([pd.DataFrame(preprocessing.transform(X),columns=X.columns),y],axis=1)
            df.to_csv(os.path.join(self.data_transformation_config.transformed_test_dir,"test.csv"),index=False)
            path_pkl=self.data_transformation_config.preprocessed_object_file_path
            with open(path_pkl,"wb") as f:
                pickle.dump(preprocessing,f)
            shutil.copy(path_pkl,ROOT_DIR)
            return DataTransformationArtifact(is_transformed=True,message="Data Transformed",transformed_train_file_path=os.path.join(self.data_transformation_config.transformed_train_dir,"train.csv"),transformed_test_file_path=os.path.join(self.data_transformation_config.transformed_test_dir,"test.csv"),preprocessed_object_file_path=self.data_transformation_config.preprocessed_object_file_path)

        except Exception as e:
            raise forest_cover_exception(e,sys) from e

    def initiate_data_transformation(self):
        try:
            data_transformation_artifact=self.get_transformer_object_and_test_train_dir()
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise forest_cover_exception(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")



