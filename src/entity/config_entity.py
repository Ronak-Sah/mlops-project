from dataclasses import dataclass
from pathlib import Path
from typing import Dict 

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path
    train_set_size: float
    test_set_size: float
    val_set_size: float

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir:Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path
    status_file_path: Path
    all_schema: Dict[str, str]

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir:Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path
    scaler: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir:Path
    booster: str
    device: str
    learning_rate: float
    max_depth: int
    early_stopping_rounds: int
    n_estimators: int

    


