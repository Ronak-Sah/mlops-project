from src.entity.config_entity import DataIngestionConfig

from src.constants import *
from src.utils.common import read_yaml,create_directories
from src.exception import CustomException
import sys

class ConfigurationManager:
    def __init__(self,config_filepath= CONFIG_FILE_PATH,params_filepath= PARAMS_FILE_PATH):

        try:
            self.config=read_yaml(config_filepath)
            self.params=read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])
        except Exception as e:
            raise CustomException(e,sys)
    

    def get_data_ingestion(self) -> DataIngestionConfig:
        try:

            config = self.config.data_ingestion 
            params = self.params.data_ingestion              
            create_directories([config.root_dir])              
            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                val_data_path=config.val_data_path,
                train_set_size=params.train_set_size,
                test_set_size=params.test_set_size,
                val_set_size=params.val_set_size
            )

            return data_ingestion_config

        except Exception as e:
            raise CustomException(e,sys)