from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir:Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path
    status_file_path: Path
    all_schema: Dict[str, str]

