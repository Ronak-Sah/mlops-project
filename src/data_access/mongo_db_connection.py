import pymongo
import pandas as pd
import sys
from src.constants import MONGO_DB_URL,COLLECTION_NAME,DB_NAME
from src.exception import CustomException
from src.logger import logging

class MongoDb():
    def __init__(self):
        try:
            self.client=pymongo.MongoClient(MONGO_DB_URL)
            self.database=self.client[DB_NAME]
            self.collection=self.database[COLLECTION_NAME]
            logging.info(f"Connected to MongoDB: {DB_NAME}")
        except Exception as e:
            logging.info("Failed to connect MongoDB")
            raise CustomException(e,sys)

    def fetch_data(self)->pd.DataFrame:
        try:
            logging.info(f"Fetching data from collection: {COLLECTION_NAME}")
            df=pd.DataFrame(list(self.collection.find()))
            if "_id" in df.columns:
                df.drop(columns=["_id"], axis=1, inplace=True)
            logging.info(f"Successfully fetched {len(df)} records from MongoDB.")
            return df
        except Exception as e:
            logging.error("e")
            raise CustomException(e,sys)
