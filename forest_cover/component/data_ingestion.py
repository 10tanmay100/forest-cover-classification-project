
from forest_cover.entity.config_entity import DataIngestionConfig
import sys,os
from forest_cover.exception import forest_cover_exception
from forest_cover.constant import *
from forest_cover.logger import logging
from forest_cover.entity.artifact_entity import DataIngestionArtifact
import numpy as np
import pandas as pd
from forest_cover.util.util import *
from sklearn.model_selection import train_test_split
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
class DataIngestion:
    
    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config
            os.makedirs(self.data_ingestion_config.raw_data_dir,exist_ok=True)
        except Exception as e:
            raise forest_cover_exception(e,sys)
    def get_data_from_database(self):
        try:
            cloud_config= {'secure_connect_bundle': 'E:\ml project\secure-connect-forest-cover-database.zip'}
            auth_provider = PlainTextAuthProvider('vdKxIwoytZzbJqZOjwPwMCCJ', '1DSXY5A7v-XsU9KZB5jMXCRtL,ieqwb_+4GDMeerEh+a,bZZD4n,qmHpv-+hUzLDDMNDTHDw0MhpteWXry-nmSoBy-zo80DkLRjavIER7-0ubcJ13BGxMqv035BtT4qx')
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect("forest_cover")

            row = session.execute("SELECT * FROM forest_cover.train").one()
            session.row_factory = pandas_factory
            session.default_fetch_size = None
            query = "SELECT * FROM forest_cover.train"
            rslt = session.execute(query, timeout=None)
            df = rslt._current_rows
            df.to_csv(os.path.join(self.data_ingestion_config.raw_data_dir,"dumped_data.csv"),index=False)
        except Exception as e:
            raise forest_cover_exception(e,sys) from e


    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
            os.makedirs(self.data_ingestion_config.ingested_test_dir,exist_ok=True)
            file_name=os.listdir(os.path.join(PATH_READ_LATEST_INGESTION_DATA,os.listdir(PATH_READ_LATEST_INGESTION_DATA)[(len(os.listdir(PATH_READ_LATEST_INGESTION_DATA)))-1],"raw_data"))[0]

            forest_file_path = os.path.join(raw_data_dir,file_name)


            logging.info(f"Reading csv file: [{forest_file_path}]")
            forest_data_frame = pd.read_csv(forest_file_path)

            x=forest_data_frame.drop("Cover_Type",axis=1)
            y=forest_data_frame["Cover_Type"]
            
            logging.info(f"Splitting data into train and test")

            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
            train_data=pd.concat([x_train,y_train],axis=1)

            test_data=pd.concat([x_test,y_test],axis=1)
            train_file_path=os.path.join(self.data_ingestion_config.ingested_train_dir,"train_data.csv")
            train_data.to_csv(train_file_path,index=False)
            test_file_path=os.path.join(self.data_ingestion_config.ingested_test_dir,"test_data.csv")
            test_data.to_csv(test_file_path,index=False)

            data_ingestion_artifact=DataIngestionArtifact(train_file_path=train_file_path,test_file_path=test_file_path,is_ingested=True,message="Data Ingested")
            return data_ingestion_artifact
        except Exception as e:
            raise forest_cover_exception(e,sys) from e

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            self.get_data_from_database()
            return self.split_data_as_train_test()
        except Exception as e:
            raise forest_cover_exception(e,sys) from e
        
    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")




