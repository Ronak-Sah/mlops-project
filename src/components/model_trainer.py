from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.logger import logging
from src.exception import CustomException
from src.configuration import ConfigurationManager
import os,sys,dagshub
import xgboost as xgb
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_error
import pandas as pd
import mlflow
import joblib,json
from pathlib import Path
import numpy as np
from src.constants import DAGSHUB_URI,MLFLOW_TRACKING_PASSWORD,MLFLOW_TRACKING_USERNAME

class ModelTrainer:


    def __init__(self,config:ModelTrainerConfig):
        """
        Input: 
        1. ModelTrainerConfig (Params like learning_rate, n_estimators)
        2. DataTransformationArtifact (Contains the paths to scaled_train.csv and scaled_test.csv)
        """
        try:
            self.config=config
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            # mlflow.set_tracking_uri("sqlite:///mlflow.db")
            logging.info("Starting Model Training")
            train_df=pd.read_csv(self.config.transformed_train_path)
            val_df=pd.read_csv(self.config.transformed_val_path)

            X_train=train_df.iloc[:,:-1]
            y_train=train_df.iloc[:,-1]
            X_val=val_df.iloc[:,:-1]
            y_val=val_df.iloc[:,-1]
            logging.info(f"Checking for NaNs in Target: {y_train.isnull().sum()}")
            logging.info(f"Checking for Inf in Target: {np.isinf(y_train).sum()}")
            logging.info(f"Target Max Value: {y_train.max()}, Min Value: {y_train.min()}")


            logging.info(f"Training XGBoost model with booster: {self.config.booster}")
            model=xgb.XGBRegressor(
                booster=self.config.booster,
                device=self.config.device,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                early_stopping_rounds=self.config.early_stopping_rounds,
                n_estimators=self.config.n_estimators
            )
            mlflow.log_params({"booster":self.config.booster,
                "learning_rate":self.config.learning_rate,
                "max_depth":self.config.max_depth,
                "n_estimators":self.config.n_estimators,
                "early_stopping_rounds":self.config.early_stopping_rounds
            })
            model.fit(
                X_train,y_train,
                eval_set=[(X_val,y_val)]
            )

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val) 
            metrics = {
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "train_rmse": root_mean_squared_error(y_train, y_train_pred),
                "train_r2": r2_score(y_train, y_train_pred),
                "val_mae": mean_absolute_error(y_val, y_val_pred),
                "val_rmse": root_mean_squared_error(y_val, y_val_pred),
                "val_r2": r2_score(y_val, y_val_pred)
            }

            mlflow.log_metrics(metrics)
            logging.info(f"Model Training Metrics: {metrics}")

            mlflow.log_metrics(metrics)
            model_info_mlflow=mlflow.xgboost.log_model(model,artifact_path="model")


            model_path = os.path.join(self.config.root_dir,"model.pkl")
            joblib.dump(model, model_path)
            metric_path=os.path.join(self.config.root_dir,"metrics.json")
            with open(metric_path,"w") as f:
                json.dump(metrics,f,indent=4)
            experiment_info_path=os.path.join(self.config.root_dir,"experiment.json")
            experiment_info={
                "run_id":mlflow.active_run().info.run_id,
                "model_path":"model",
                "model_uri":model_info_mlflow.model_uri
            }
            with open(experiment_info_path,"w") as f:
                json.dump(experiment_info,f,indent=4)
            return ModelTrainerArtifact(
                trained_model_path=model_path,
                metric_path=metric_path,
                experiment_info_path=experiment_info_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    try:
        logging.info("Stage: Model Training started")
        config = ConfigurationManager()
        model_training_config = config.get_model_trainer()

        mlflow.set_tracking_uri(DAGSHUB_URI)
        dagshub.init(repo_owner='ronaksah75', repo_name='mlops-project', mlflow=True)
        os.environ['MLFLOW_TRACKING_USERNAME']=MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD']=MLFLOW_TRACKING_PASSWORD

        mlflow_path=Path("artifacts/runs")

        with open(os.path.join(mlflow_path,"parent_run_id.txt"),"r") as f:
            parent_id=f.read().strip()
        mlflow.set_experiment("House_Price_Prediction")
        with mlflow.start_run(run_id=parent_id):
            with mlflow.start_run(run_name="Model_Training_Step", nested=True) as child:
                model_training=ModelTrainer(model_training_config)
                model_training.initiate_model_trainer()
        logging.info("Stage: Model Training completed ")
    except Exception as e:
        # logging.exception(e)
        raise CustomException(e,sys)  