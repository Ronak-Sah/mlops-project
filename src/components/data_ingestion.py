from src.data_access.mongo_db_connection import MongoDb
from src.entity.config_entity import DataIngestionConfig
from src.logger import logging
from src.exception import CustomException
from src.configuration import ConfigurationManager
from src.utils.common import create_directories
import pandas as  pd
import numpy as np
import os
import sys,dagshub
import mlflow
from pathlib import Path
from src.entity.artifact_entity import DataIngestionArtifact
from src.constants import *
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
            
            # with mlflow.start_run(nested=True,run_name="Data_Ingestion_Step"):

            logging.info("Exporting data from MongoDB to Dataframe")
            df=self.mongo_db.fetch_data()

            if df.empty:
                logging.warning("Fetched DataFrame is empty. Check MongoDB collection.")
                return None
                
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

            return DataIngestionArtifact(
                train_data_path=train_file,
                test_data_path=test_file,
                val_data_path=val_file
            )
        except Exception as e:
            logging.error("Failed to execute load_data")
            raise CustomException(e,sys)
        



if __name__ == "__main__":
    try:

        logging.info("Stage: Data Ingestion started")
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion()

        os.environ['MLFLOW_TRACKING_USERNAME']=MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD']=MLFLOW_TRACKING_PASSWORD
        mlflow.set_tracking_uri(DAGSHUB_URI)
        # dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

        # mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("House_Price_Prediction")
        with mlflow.start_run(run_name="Full_Pipeline_Parent") as parent_run:
            with mlflow.start_run(run_name="Data_Ingestion_Step", nested=True):
                mlflow_path=Path("artifacts/runs")
                run_id = parent_run.info.run_id

                os.makedirs(mlflow_path,exist_ok=True)
                with open(os.path.join(mlflow_path,"parent_run_id.txt"),"w") as f:
                    f.write(run_id)

                data_ingestion = DataIngestion(data_ingestion_config)
                data_ingestion.load_data()
        logging.info("Stage: Data Ingestion completed")
    except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)


