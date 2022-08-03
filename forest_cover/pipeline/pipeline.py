from forest_cover.config.configuration import Configuration
from forest_cover.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,ModelTrainerConfig
from forest_cover.logger import logging
from forest_cover.component.data_ingestion import DataIngestion
from forest_cover.component.data_validation import DataValidation
from forest_cover.component.data_transformation import DataTransformation
from forest_cover.component.model_trainer import ModelTrainer
from forest_cover.entity.artifact_entity import *
from forest_cover.exception import forest_cover_exception
from forest_cover.constant import *
from forest_cover.util.util import *

class Pipeline:
    def __init__(self, config: Configuration ) -> None:
        self.config=config()
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise forest_cover_exception(e, sys) from e
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) \
            -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact
                                             )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise forest_cover_exception(e, sys) from e

    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact
                                  ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise forest_cover_exception(e, sys)

    def start_model_training(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer=ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_training()
        except Exception as e:
            raise forest_cover_exception(e,sys) from e



    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact=self.start_model_training(data_transformation_artifact=data_transformation_artifact)

            return "done"
        except Exception as e:
            raise forest_cover_exception(e,sys) from e
        




