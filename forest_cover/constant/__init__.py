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

NUMERICAL_COLUMNS=['Aspect', 'Elevation','Hillshade_9am',
       'Hillshade_Noon', 'Horizontal_Distance_To_Fire_Points',
       'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Slope','Soil_Type10','Soil_Type29','Soil_Type3','Soil_Type4','Soil_Type23','Vertical_Distance_To_Hydrology', 'Wilderness_Area1','Wilderness_Area2', 'Wilderness_Area3']

# Model Training related variables

MODEL_TRAINER_ARTIFACT_DIR = "model_trainer"
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY_0 = "model_file_name_cluster0"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY_1 = "model_file_name_cluster1"
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"



LOGISTICS_PARAMS_TUNING={"penalty":['l1', 'l2', 'elasticnet'],"tol":[1e-3,1e-4,1e-5,1e-6],"solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],"max_iter":[100,200,300],"multi_class":["ovr", "multinomial"]}

SVC_PARAMS_TUNING={"kernel":["linear", "poly", "rbf", "sigmoid"],"degree":[1,2,3],"gamma":["scale", "auto"],"tol":[1e-2,1e-2,1e-3],"decision_function_shape":["ovo", "ovr"]}

DECISION_TREE_TUNING={"criterion":["gini", "entropy", "log_loss"],"max_depth":[10,20,40,50,100,200,400,600],"splitter":["best", "random"],"min_samples_split":range(2,50),"max_features":['auto', 'sqrt', 'log2']}

RANDOM_FOREST_TUNING={"criterion":["gini", "entropy", "log_loss"],"max_depth":range(6,15),"min_samples_split":[2,3,4,5,10,15,20],"max_features":["sqrt", "log2"]}

KNN_TUNING={"n_neighbors":[5,6,7,8,9,10],"weights":["uniform", "distance"],"algorithm":["auto", "ball_tree", "kd_tree", "brute"],"leaf_size":[5,10,15,20,25,30,35,40,45,50,55,60,65,70]}

GD_TUNING={"loss":["log_loss", "deviance", "exponential"],"learning_rate":[0.1,0.001,0.0001,0.02,0.05,0.09,0.006,0.7,0.8,0.9],"n_estimators":[100,200,300,400],"criterion":["friedman_mse", "squared_error", "mse"]}