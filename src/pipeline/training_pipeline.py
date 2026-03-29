from src.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys,mlflow

class TrainPipeline:
    def __init__(self):
        self.config=ConfigurationManager()
       

    def start_data_ingestion(self):
        try:
            # mlflow.set_tracking_uri("sqlite:///mlflow.db")
            # mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_tracking_uri("file:///D:\Ml Dl\Project\mlops-project\mlruns")
            mlflow.set_experiment("1-Run") 
            print(mlflow.get_experiment_by_name("Default"))
        
            with mlflow.start_run(run_name="Full_Pipeline_Execution") as run:
                logging.info("Entered the data_ingestion component of TrainPipeline class")
                # data_ingestion_config=self.config.get_data_ingestion()
                # data_ingestion=DataIngestion(data_ingestion_config)
                # data_ingestion.load_data()
                logging.info("Entered the data_validation component of TrainPipeline class")
                data_validation=DataValidation(self.config.get_data_validation())
                data_validation.initiate_data_validation()
                logging.info("Entered the data_transformation component of TrainPipeline class")
                data_transformation=DataTransformation(self.config.get_data_tranformation())
                data_transformation_artifacts=data_transformation.initiate_data_transformation()
                logging.info("Entered the model_trainer component of TrainPipeline class")
                model_trainer=ModelTrainer(self.config.get_model_trainer(),data_transformation_artifacts)
                model_trainer.initiate_model_trainer()

        except Exception as e:
            raise CustomException(e,sys)

