import os,sys
from src.logger import logging
from src.exception import CustomException
import json,mlflow,dagshub
from src.constants import DAGSHUB_URI,MLFLOW_TRACKING_PASSWORD,MLFLOW_TRACKING_USERNAME

class ModelRegistery:
    def __init__(self):
        self.info_path=os.path.join("artifacts","model_trainer","experiment.json")
        self.scaler_info_path=os.path.join("artifacts","data_transformation","experiment.json")
    def load_model_info(self,path):
        try:
            with open(path,"r") as r:
                model_info=json.load(r)
            logging.info(f"Model info loaded from {path}")
            return model_info
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    def register_model(self,model_name:str):
        try:

            model_info=self.load_model_info(self.info_path)
            model_uri=model_info["model_uri"]
            print(model_uri)
            logging.info(f"Registering model from URI: {model_uri}")
            model_version = mlflow.register_model(model_uri,model_name)

            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            logging.info("Model registered")
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        
    
    def register_scaler(self,model_name):
        try:

            model_info=self.load_model_info(self.scaler_info_path)
            model_uri=model_info["model_uri"]
            logging.info(f"Registering sclaer model from URI: {model_uri}")
            model_version = mlflow.register_model(model_uri, model_name)

            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            logging.info("Sclaer Model registered")

        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    try:
        logging.info("Stage: Model registering started")

        mlflow.set_tracking_uri(DAGSHUB_URI)
        dagshub.init(repo_owner='ronaksah75', repo_name='mlops-project', mlflow=True)
        os.environ['MLFLOW_TRACKING_USERNAME']=MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD']=MLFLOW_TRACKING_PASSWORD

        model_register=ModelRegistery()
        model_register.register_model("house_price")
        model_register.register_scaler("scaler_model")

    except Exception as e:
        logging.error(e)
        raise CustomException(e,sys)