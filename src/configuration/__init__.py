from src.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from src.entity.config_entity import ModelTrainerConfig
from src.constants import *
from src.utils.common import read_yaml,create_directories
from src.exception import CustomException
import sys

class ConfigurationManager:
    def __init__(self,config_filepath= CONFIG_FILE_PATH,params_filepath= PARAMS_FILE_PATH,schema_filepath=SCHEMA_FILE_PATH):

        try:
            self.config=read_yaml(config_filepath)
            self.params=read_yaml(params_filepath)
            self.schema=read_yaml(schema_filepath)
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
        


    def get_data_validation(self) -> DataValidationConfig:
        try:
            config=self.config.data_validation
            schema=self.schema.data_validation
            create_directories([config.root_dir]) 
            data_validation_config=DataValidationConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                val_data_path=config.val_data_path,
                all_schema=schema.columns,
                status_file_path=config.status_file_path
            )

            return data_validation_config
        except Exception as e:
            raise CustomException(e,sys)
        

    def get_data_tranformation(self)->DataTransformationConfig:
        try:
            config=self.config.data_transformation
            params=self.params.data_transformation
            create_directories([config.root_dir])

            data_transformation_config=DataTransformationConfig(
                root_dir=config.root_dir,
                train_data_path=config.train_data_path,
                test_data_path=config.test_data_path,
                val_data_path=config.val_data_path,
                scaler=params.scaler 
            )
            
            return data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
    


    def get_model_trainer(self)->ModelTrainerConfig:
        try:
            config=self.config
            params=self.params
            create_directories([config.root_dir])

            model_trainer_config=ModelTrainerConfig(
                root_dir=config.root_dir,
                epochs=params.epochs
            )
            return model_trainer_config
        except Exception as e:
            raise CustomException(e,sys)
