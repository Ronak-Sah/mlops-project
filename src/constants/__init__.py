from pathlib import Path
from dotenv import  load_dotenv
import os

load_dotenv()

MONGO_DB_URL=os.getenv("mongo_db_url")
DB_NAME='House_Price'
COLLECTION_NAME='House_Price_Prediction'

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")