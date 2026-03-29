from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.logger import logging
from src.exception import CustomException
import os,sys
import xgboost as xgb
from sklearn.metrics import r2_score,root_mean_squared_error,mean_absolute_error
import pandas as pd
import mlflow
import joblib
from pathlib import Path
import numpy as np

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig,artifact:DataTransformationArtifact):
        """
        Input: 
        1. ModelTrainerConfig (Params like learning_rate, n_estimators)
        2. DataTransformationArtifact (Contains the paths to scaled_train.csv and scaled_test.csv)
        """
        try:
            self.config=config
            self.artifact=artifact
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            # mlflow.set_tracking_uri("sqlite:///mlflow.db")
            logging.info("Starting Model Training")
            train_df=pd.read_csv(self.artifact.transformed_train_path)
            val_df=pd.read_csv(self.artifact.transformed_val_path)

            X_train=train_df.iloc[:,:-1]
            y_train=train_df.iloc[:,-1]
            X_val=val_df.iloc[:,:-1]
            y_val=val_df.iloc[:,-1]
            logging.info(f"Checking for NaNs in Target: {y_train.isnull().sum()}")
            logging.info(f"Checking for Inf in Target: {np.isinf(y_train).sum()}")
            logging.info(f"Target Max Value: {y_train.max()}, Min Value: {y_train.min()}")

            with mlflow.start_run(nested=True, run_name="Model_Training_Step"):
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
                mlflow.xgboost.log_model(model,name="model")


                model_path = os.path.join(self.config.root_dir,"model.pkl")
                joblib.dump(model, model_path)

                return ModelTrainerArtifact(
                    trained_model_path=Path(model_path)
                )

        except Exception as e:
            raise CustomException(e,sys)