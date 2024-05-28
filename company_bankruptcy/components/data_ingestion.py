import pandas as pd
import numpy as np

from company_bankruptcy.logger.logger import logging
from company_bankruptcy.exception.exception import CustomException
from company_bankruptcy.data_access.mongo_db_connection import MongoOps
from company_bankruptcy.constants.constants import DATABASE_NAME, COLLECTION_NAME, MONGODB_COLLECTION_STR

import os
import sys
from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

MONGODB_COLLECTION_STR = "mongodb+srv://vcharchian:12DyeUWoTDa10AJn@cluster0.xbq0vxb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts', 'data.csv')
    train_data_path:str = os.path.join('artifacts', 'train_data.csv')
    test_data_path:str = os.path.join('artifacts', 'test_data.csv')

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion started')
        try:
            logging.info('Reading the raw data')
            mongo_instance = MongoOps(
                client_url=MONGODB_COLLECTION_STR
            )
            data = mongo_instance.get_records(coll_name=COLLECTION_NAME, db_name=DATABASE_NAME)
            logging.info('Data loaded')
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            logging.info('Saving the data')
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Data saved')
            logging.info('Splitting the data into train and test sets')
            train_df, test_df = train_test_split(
                data, 
                test_size=0.1, 
                random_state=13, 
                stratify=data['Bankrupt?']
            )
            logging.info('Saving train and test sets')
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info('Sets are saved')
            logging.info('Data ingestion completed')
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            logging.info('Error occured during data ingestion')
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_ingestion_obj = DataIngestion()
    train_path, test_path = data_ingestion_obj.initiate_data_ingestion()
