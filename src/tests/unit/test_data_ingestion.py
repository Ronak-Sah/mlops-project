import pytest
from src.components.data_ingestion import DataIngestion
import os
import pandas as pd
class TestDataIngestion:
    @pytest.fixture
    def setup_ingstion(self):
        return DataIngestion()
    

    def test_init_ingestion(self,setup_ingstion):
        ingestion=setup_ingstion.load_data()

        assert ingestion.train_data_path==os.path.join('artifacts','data_ingestion','train','train.csv')
        assert ingestion.test_data_path==os.path.join('artifacts','data_ingestion','test','test.csv')
        assert ingestion.val_data_path==os.path.join('artifacts','data_ingestion','val','val.csv')

    def test_initiate_data_ingestion(self,setup_ingstion):
        ingestion=setup_ingstion.load_data()

        assert os.path.exists(ingestion.train_data_path)
        assert os.path.exists(ingestion.test_data_path)
        assert os.path.exists(ingestion.val_data_path)

        df_train=pd.read_csv(ingestion.train_data_path)
        df_test=pd.read_csv(ingestion.test_data_path)
        df_val=pd.read_csv(ingestion.val_data_path)

        assert len(df_train)>0
        assert len(df_test)>0
        assert len(df_val)>0
    

    
                   
