from src.data_access.mongo_db_connection import MongoDb
from src.entity.config_entity import DataIngestionConfig
from src.logger import logging
from src.exception import CustomException
from src.utils.common import create_directories
import pandas as  pd
import numpy as np
import os
import sys
import mlflow

class DataIngestion():
    def __init__(self,config:DataIngestionConfig):
        try:
            self.config=config
            self.mongo_db=MongoDb()
        except Exception as e:
            logging.error("Failed to initialize Data Ingestion component")
            raise CustomException(e,sys)


    def load_data(self):
        try:
            mlflow.set_tracking_uri("file:///D:/Ml Dl/Project/mlops-project/mlruns")
            mlflow.set_experiment("Data Ingestion")
            with mlflow.start_run():

                logging.info("Exporting data from MongoDB to Dataframe")
                df=self.mongo_db.fetch_data()

                if df.empty:
                    logging.warning("Fetched DataFrame is empty. Check MongoDB collection.")
                    return
                
                shuffled_indices=np.random.permutation(len(df))

                create_directories([self.config.train_data_path,self.config.test_data_path,self.config.val_data_path])

                test_size=int(len(df)*self.config.test_set_size)
                val_size=int(len(df)*self.config.val_set_size)

                mlflow.log_param("train_split_ratio",self.config.train_set_size)
                mlflow.log_param("test_split_ratio",self.config.test_set_size)
                mlflow.log_param("val_split_ratio",self.config.val_set_size)

                test_indices=shuffled_indices[:test_size]
                val_indices=shuffled_indices[test_size:val_size+test_size]
                train_indices=shuffled_indices[val_size+test_size:]

                train_data=df.iloc[train_indices]
                test_data=df.iloc[test_indices]
                val_data=df.iloc[val_indices]

                train_file = os.path.join(self.config.train_data_path, "train.csv")
                test_file = os.path.join(self.config.test_data_path, "test.csv")
                val_file = os.path.join(self.config.val_data_path, "val.csv")

                train_data.to_csv(train_file,index=False)
                test_data.to_csv(test_file,index=False)
                val_data.to_csv(val_file,index=False)

                mlflow.log_artifact(train_file, artifact_path="ingested_train_data")
                mlflow.log_artifact(test_file,artifact_path="ingested_test_path")

                logging.info(f"Training data saved at folder - {self.config.train_data_path}")
                logging.info(f"Testing data saved at folder - {self.config.test_data_path}")
                logging.info(f"Validation data saved at folder - {self.config.val_data_path}")
        except Exception as e:
            logging.error("Failed to execute load_data")
            raise CustomException(e,sys)


