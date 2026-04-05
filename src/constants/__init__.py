from pathlib import Path
from dotenv import  load_dotenv
import os

load_dotenv(override=False)

MONGO_DB_URL=os.getenv("mongo_db_url")
if MONGO_DB_URL is None:
    print("Warning: MONGO_DB_URL not found in environment variables!")
DB_NAME='House_Price'
COLLECTION_NAME='House_Price_Prediction'

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")
SCHEMA_FILE_PATH = Path("config/schema.yaml")

DAGSHUB_URI=os.getenv("dagshub_uri")
MLFLOW_TRACKING_USERNAME=os.getenv("mlflow_tracking_username")
MLFLOW_TRACKING_PASSWORD=os.getenv("mlflow_tracking_password")

REPO_OWNER="ronaksah75"
REPO_NAME="mlops-project"

