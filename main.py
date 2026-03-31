from src.pipeline.training_pipeline import TrainPipeline
import mlflow
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# # mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("1-Run") 
print(mlflow.get_experiment_by_name("Default"))
with mlflow.start_run(run_name="Full_Pipeline_Execution") as run:

    pipeline=TrainPipeline()
    data_ingest_artifact=pipeline.start_data_ingestion()
    data_val_artifact=pipeline.start_data_validation()
    data_trans_artifact=pipeline.start_data_transformation()
    model_train_artifact=pipeline.start_model_trainer()