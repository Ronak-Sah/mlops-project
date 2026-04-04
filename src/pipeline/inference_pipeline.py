from dataclasses import dataclass
import pandas as pd
import joblib,os,mlflow,dagshub,sys
from src.entity.config_entity import InferenceConfig
from src.constants import DAGSHUB_URI,MLFLOW_TRACKING_PASSWORD,MLFLOW_TRACKING_USERNAME
from src.logger import logging
from src.exception import CustomException

class ModelPredict:
    def __init__(self,config:InferenceConfig):
        self.config=config
        mlflow.set_tracking_uri(DAGSHUB_URI)
        dagshub.init(repo_owner='ronaksah75', repo_name='mlops-project', mlflow=True)
        os.environ['MLFLOW_TRACKING_USERNAME']=MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD']=MLFLOW_TRACKING_PASSWORD
    
    def get_latest_model_version(self,model_name:str):
        try:

            client=mlflow.MlflowClient()
            latest_version=client.get_latest_versions(model_name,stages=["Staging"])
            if not latest_version:
                latest_version=client.get_latest_versions(model_name,stages=["None"])
            logging.info(f"fetched {model_name} of version {latest_version[0].version}")
            return latest_version[0].source 
        except Exception as e:
            logging.exception(e)
            raise CustomException(e,sys)
        

    def predict(self):
        model_uri=self.get_latest_model_version("house_price")
        model=mlflow.pyfunc.load_model(model_uri)
        scaler_uri=self.get_latest_model_version("scaler_model")
        scaler=mlflow.sklearn.load_model(scaler_uri)
        df=pd.DataFrame({
            "bedrooms":[self.config.bedrooms],"bathrooms":[self.config.bathrooms],
            "sqft_living": [self.config.sqft_living],"sqft_lot": [self.config.sqft_lot],
            "floors": [self.config.floors],"waterfront": [self.config.waterfront],
            "view": [self.config.view],"condition": [self.config.condition],"grade": [self.config.grade],
            "sqft_above": [self.config.sqft_above],"sqft_basement": [self.config.sqft_basement],
            "yr_built": [self.config.yr_built],"yr_renovated": [self.config.yr_renovated],
            "zipcode": [self.config.zipcode],"lat": [self.config.lat],"long": [self.config.long],
            "sqft_living15": [self.config.sqft_living15]
        })
        df=df.astype(float)
        scaled_df=scaler.transform(df)
        y=model.predict(scaled_df)

        return y[0]
    

