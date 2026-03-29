import pandas as pd
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer
import os,sys
import joblib
from src.exception import CustomException
from src.logger import logging
from pathlib import Path
import mlflow
class DataTransformation:
    def  __init__(self,config:DataTransformationConfig):
        try:
            self.config=config
            all_scaler_types={"StandardScaler":StandardScaler(),
                "MinMaxScaler":MinMaxScaler(),
                "MaxAbsScaler":MaxAbsScaler(),
                "Normalizer":Normalizer()
            }
            
            self.scaler=all_scaler_types.get(self.config.scaler)
            if self.scaler is None:
                logging.info(f"Scaler {self.config.scaler} not found in supported scalers")
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self):
        try:
            with mlflow.start_run(nested=True,run_name="Data_Transformation_Step"):
                logging.info("Starting data transformation")
                train_df=pd.read_csv(self.config.train_data_path)
                val_df=pd.read_csv(self.config.val_data_path)
                test_df=pd.read_csv(self.config.test_data_path)

                train_file_path = os.path.join(self.config.root_dir,"train.csv")
                test_file_path = os.path.join(self.config.root_dir, "test.csv")
                val_file_path = os.path.join(self.config.root_dir, "val.csv")
                scaler_file_path = os.path.join(self.config.root_dir,"scaler.pkl")

                train_df.dropna(subset=[train_df.columns[-1]], inplace=True)
                test_df.dropna(subset=[test_df.columns[-1]], inplace=True)
                val_df.dropna(subset=[val_df.columns[-1]], inplace=True)

                target_column= train_df.columns[-1]
                def transform_df(df, fit=False):
                    X=df.drop(columns=[target_column])
                    y=df[[target_column]]
                    
                    if fit:
                        X_scaled=self.scaler.fit_transform(X)
                    else:
                        X_scaled =self.scaler.transform(X)
                    
                    
                    transformed_df = pd.DataFrame(X_scaled, columns=X.columns)
                    transformed_df[target_column] = y.values
                    return transformed_df

                logging.info(f"Applying {self.config.scaler} transformation")

                mlflow.log_param("scaler_type",self.config.scaler)
                mlflow.sklearn.log_model(self.scaler,name="scaler_model")
                train_transformed = transform_df(train_df, fit=True)
                val_transformed = transform_df(val_df)
                test_transformed = transform_df(test_df)

                train_transformed.to_csv(train_file_path,index=False)
                val_transformed.to_csv(val_file_path,index=False)
                test_transformed.to_csv(test_file_path,index=False)
                joblib.dump(self.scaler,scaler_file_path)

                return DataTransformationArtifact(
                    transformed_train_path=Path(train_file_path),
                    transformed_test_path=Path(test_file_path),
                    transformed_val_path=Path(val_file_path),
                    scaler_file_path=Path(scaler_file_path)
                )
        except Exception as e:
            raise CustomException(e,sys)     


