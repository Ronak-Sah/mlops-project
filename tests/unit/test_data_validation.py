import pytest
from src.components.data_validation import DataValidation
import os
import pandas as pd
class TestDataValidation:
    @pytest.fixture
    def setup_validation(self):
        return DataValidation()
    

    def test_init_validation(self,setup_validation):
        validation=setup_validation.initiate_data_validation()

        assert validation
        
   
    

    
                   
