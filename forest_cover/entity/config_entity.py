from collections import namedtuple

DataIngestionConfig=namedtuple("DataIngestionConfig",
["raw_data_dir","ingested_train_dir","ingested_test_dir"])

#DataValidationConfig
DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path","report_file_path","report_page_file_path"])


DataTransformationConfig=namedtuple("DataTransformationConfig",
["transformed_train_dir","transformed_test_dir","preprocessed_object_folder_path",
"preprocessed_object_file_path"])

#ModelTrainerConfig
ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_model_file_path_cluster_folder","trained_model_file_path_cluster0","trained_model_file_path_cluster1","base_accuracy"])



#datapipelineconfig
TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])