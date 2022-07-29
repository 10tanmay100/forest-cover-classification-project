import os
from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

os.chdir(r"E:\Ineuron\Project\forest cover classification project\forest-cover-classification-project")
ROOT_DIR=os.getcwd()
CURRENT_TIME_STAMP = get_current_time_stamp()
CONFIG_FILE_PATH="E:\\Ineuron\\Project\\forest cover classification project\\forest-cover-classification-project\\config\\config.yaml"

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"
# Data Ingestion
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"

PATH_READ_LATEST_INGESTION_DATA="E:\\Ineuron\\Project\\forest cover classification project\\forest-cover-classification-project\\forest_cover\\artifact\\data_ingestion"
ARTIFACT_DIRECTORY="E:\\Ineuron\\Project\\forest cover classification project\\forest-cover-classification-project\\forest_cover"

# Data Validation related variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_ARTIFACT_DIR_NAME="data_validation"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"

SCHEMA_FILE_PATH="E:\\Ineuron\\Project\\forest cover classification project\\forest-cover-classification-project\\config\\schema.yaml"
DATA_INGESTION_LATEST_TRAIN_DIR="E:\\Ineuron\\Project\\forest cover classification project\\forest-cover-classification-project\\forest_cover\\artifact\\data_ingestion"
DATA_INGESTION_LATEST_TEST_DIR="E:\\Ineuron\\Project\\forest cover classification project\\forest-cover-classification-project\\forest_cover\\artifact\\data_ingestion"



# Data Transformation related variables
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY = "preprocessed_object_file_name"

NUMERICAL_COLUMNS=['Aspect', 'Elevation', 'Hillshade_3pm', 'Hillshade_9am',
       'Hillshade_Noon', 'Horizontal_Distance_To_Fire_Points',
       'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Slope', 'Soil_Type1', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type2',
       'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
       'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
       'Soil_Type28', 'Soil_Type29', 'Soil_Type3', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type4', 'Soil_Type40', 'Soil_Type5', 'Soil_Type6', 'Soil_Type9','Vertical_Distance_To_Hydrology', 'Wilderness_Area1','Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']

# Model Training related variables

MODEL_TRAINER_ARTIFACT_DIR = "model_trainer"
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"






