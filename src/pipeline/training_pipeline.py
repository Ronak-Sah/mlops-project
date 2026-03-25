from src.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
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

