from dataclasses import dataclass
from pathlib import Path

# Configuration for data ingestion

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path
    train_set_size: float
    test_set_size: float
    val_set_size: float

