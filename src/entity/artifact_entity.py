from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifact:
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path


@dataclass(frozen=True)
class DataTransformationArtifact:
    transformed_train_path: Path
    transformed_test_path: Path
    transformed_val_path: Path
    scaler_file_path:Path

@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_path: Path
    metric_path: Path