import pytest
from src.components.data_transformation import DataTransformation
import os
import pandas as pd
class TestDataTransformation:
    @pytest.fixture
    def setup_transformation(self):
        return DataTransformation()
    

    def test_init_transformation(self,setup_transformation):
        transformation=setup_transformation.initiate_data_transformation()

        assert transformation.transformed_train_path==os.path.join('artifacts','data_transformation','train.csv')
        assert transformation.transformed_test_path==os.path.join('artifacts','data_transformation','test.csv')
        assert transformation.transformed_val_path==os.path.join('artifacts','data_transformation','val.csv')
        assert transformation.scaler_file_path==os.path.join('artifacts','data_transformation','scaler.pkl')

    def test_initiate_data_ingestion(self,setup_transformation):
        transformation=setup_transformation.initiate_data_transformation()

        assert os.path.exists(transformation.transformed_train_path)
        assert os.path.exists(transformation.transformed_test_path)
        assert os.path.exists(transformation.transformed_val_path)


        df_train=pd.read_csv(transformation.transformed_train_path)
        df_test=pd.read_csv(transformation.transformed_test_path)
        df_val=pd.read_csv(transformation.transformed_val_path)

        assert len(df_train)>0
        assert len(df_test)>0
        assert len(df_val)>0
    

    
                   
