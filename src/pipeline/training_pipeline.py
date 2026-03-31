from src.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys

class TrainPipeline:
    def __init__(self):
        self.config=ConfigurationManager()
    
    def start_data_ingestion(self):
        try:
            logging.info("Entered the data_ingestion component of TrainPipeline class")
            data_ingestion_config=self.config.get_data_ingestion()
            data_ingestion=DataIngestion(data_ingestion_config)
            data_ingestion.load_data()
                    
        except Exception as e:
            raise CustomException(e,sys)

    def start_data_validation(self):
        try:
            logging.info("Entered the data_validation component of TrainPipeline class")
            data_validation=DataValidation(self.config.get_data_validation())
            data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e,sys)
    
    def start_data_transformation(self):
        try:
            logging.info("Entered the data_transformation component of TrainPipeline class")
            data_transformation=DataTransformation(self.config.get_data_transformation())
            data_transformation_artifacts=data_transformation.initiate_data_transformation()
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_trainer(self):
        try:
            logging.info("Entered the model_trainer component of TrainPipeline class")
            model_trainer=ModelTrainer(self.config.get_model_trainer())
            model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CustomException(e,sys)