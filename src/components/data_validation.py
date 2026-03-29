from src.entity.config_entity import DataValidationConfig
from src.exception import CustomException
from src.logger import logging
from datetime import datetime
import pandas as pd
import os,sys,json

class DataValidation:
    def __init__(self,config:DataValidationConfig):
        try:
            self.config=config
            self.schema=self.config.all_schema
        except Exception as e:
            raise CustomException(e,sys)

    def validate_all_files_exist(self):
        """Checks if all required ingestion artifacts exist on disk."""
        try:
            validation_status = True
            all_files = [
                self.config.train_data_path,
                self.config.test_data_path,
                self.config.val_data_path
            ]
            
            for file_path in all_files:
                if not os.path.exists(file_path):
                    validation_status = False
                    logging.warning(f"File missing: {file_path}")
            
            return validation_status

        except Exception as e:
            raise CustomException(e,sys)

    def validate_columns(self):
        """Validates number of columns and column names against schema."""
        df = pd.read_csv(self.config.train_data_path)
        validation_status = True
        
        if len(df.columns) != len(self.schema):
            validation_status = False
            logging.error(f"Column count mismatch! Schema: {len(self.schema)}, Data: {len(df.columns)}")

        for col in df.columns:
            if col not in self.schema.keys():
                validation_status = False
                logging.error(f"Column '{col}' not found in schema.yaml")
            
        return validation_status


    def initiate_data_validation(self):
        try:
            logging.info("Starting Data Validation")

            validation_report = {
                "file_existence": self.validate_all_files_exist(),
                "schema_check": self.validate_columns(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            status=self.validate_all_files_exist()
            if status:
                status=self.validate_columns()
            
            validation_report["overall_report"]=status

            with open(self.config.status_file_path,"w") as f:
                json.dump(validation_report,f,indent=4)
            
            logging.info(f"Validation Report saved to {self.config.status_file_path}")
            return status
        

        except Exception as e:
            raise CustomException(e, sys)