import pandas as pd
import numpy as np
from company_bankruptcy.logger import logging
from company_bankruptcy.exception import CustomException

import os
import sys
from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

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
            data = #ingest from the database you need mongodb data loading function
            logging.info('Data loaded')
            os.makedirs(os.path.dirname(os.join.path(self.ingestion_config.raw_data_path)), exist_ok=True)
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
            logging.info()
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
