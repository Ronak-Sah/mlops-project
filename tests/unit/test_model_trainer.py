import pytest
from src.components.model_trainer import ModelTrainer
import os
from src.configuration import ConfigurationManager
import pandas as pd
class TestModelTrainer:
    @pytest.fixture
    def setup_trainer(self):
        self.config=ConfigurationManager()
        return ModelTrainer(self.config.get_model_trainer())
    

    def test_init_trainer(self,setup_trainer):
        trainer=setup_trainer.initiate_model_trainer()

        assert trainer.trained_model_path==os.path.join('artifacts','model_trainer','model.pkl')
        assert trainer.metrics_path==os.path.join('artifacts','model_trainer','metrics.json')
        

    def test_initiate_model_trainer(self,setup_trainer):
        trainer=setup_trainer.initiate_model_trainer()

        assert os.path.exists(trainer.trained_model_path)
        assert os.path.exists(trainer.metric_path)


    

    
                   
