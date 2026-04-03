from src.pipeline.training_pipeline import TrainPipeline
import os
import pytest
import shutil

class TestPipeline:
    @pytest.fixture
    def clean_artifacts(self):
        yield
        if os.path.exists("artifacts"):
            shutil.rmtree("artifacts")

    def test_pipeline(self):

        try:
            pipeline=TrainPipeline()
            ingestion=pipeline.start_data_ingestion()
        except Exception as e:
            pytest.fail(f"Pipeline Stage 01 (Ingestion) failed: {e}")

        assert os.path.exists(ingestion.train_data_path)
        assert os.path.exists(ingestion.test_data_path)
        assert os.path.exists(ingestion.val_data_path)

        try:
            validation=pipeline.start_data_validation()
        except Exception as e:
            pytest.fail(f"Pipeline Stage 02 (Validation) failed: {e}")

        assert validation

        try:
            transformation=pipeline.start_data_transformation()
        except Exception as e:
            pytest.fail(f"Pipeline Stage 03 (Transformation) failed: {e}")

        assert os.path.exists(transformation.transformed_train_path)
        assert os.path.exists(transformation.transformed_test_path)
        assert os.path.exists(transformation.transformed_val_path)

        try:
            trainer=pipeline.start_model_trainer()
        except Exception as e:
            pytest.fail(f"Pipeline Stage 04 (Training) failed: {e}")
        
        assert os.path.exists(trainer.trained_model_path)
        assert os.path.exists(trainer.metric_path)