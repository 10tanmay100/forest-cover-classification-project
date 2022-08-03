from forest_cover.exception import forest_cover_exception
import sys
from forest_cover.logger import logging
from typing import List
import pandas as pd
import numpy as np
import shutil
import pickle
from forest_cover.constant import *
from forest_cover.util.util import *
from forest_cover.entity.config_entity import *
from forest_cover.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise forest_cover_exception(e, sys) from e
    
    def start_training_model(self):
        try:
            os.makedirs(self.model_trainer_config.trained_model_file_path_cluster_folder,exist_ok=True)
            #working on training data
            train_df=pd.read_csv(self.data_transformation_artifact.transformed_train_file_path)
            logging.info(f"Reading the file {train_df}")
            test_df=pd.read_csv(self.data_transformation_artifact.transformed_test_file_path)
            X_cluster_0,y_cluster_0=do_train_0(train_df)
            X_cluster_1,y_cluster_1=do_train_0_train(train_df)
            x_test_cluster0,y_test_cluster0=do_train_0_test(test_df)
            model_cluster0=model_tuning_1(X_cluster_0,y_cluster_0,0.5,x_test_cluster0,y_test_cluster0)
            x_test_cluster1,y_test_cluster1=do_train_1_test(test_df)
            model_cluster1=model_tuning_2(X_cluster_1,y_cluster_1,0.5,x_test_cluster1,y_test_cluster1)


            with open(self.model_trainer_config.trained_model_file_path_cluster0,'wb') as f:
                pickle_file = pickle.dump(model_cluster0,f)
            shutil.copy(self.model_trainer_config.trained_model_file_path_cluster0,ROOT_DIR)

            with open(self.model_trainer_config.trained_model_file_path_cluster1,'wb') as f:
                pickle_file = pickle.dump(model_cluster1,f)
            shutil.copy(self.model_trainer_config.trained_model_file_path_cluster1,ROOT_DIR)

            return ModelTrainerArtifact(is_trained=True,message="Training has been completed!!",trained_model_file_path_cluster0=self.model_trainer_config.trained_model_file_path_cluster0,trained_model_file_path_cluster1=self.model_trainer_config.trained_model_file_path_cluster1)

        except Exception as e:
           raise forest_cover_exception(e,sys) from e
    
    def initiate_model_training(self):
        self.start_training_model()

