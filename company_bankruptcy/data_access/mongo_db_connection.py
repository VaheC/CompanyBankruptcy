import pandas as pd
import pymongo
import json

from company_bankruptcy.exception.exception import CustomException
from company_bankruptcy.logger.logger import logging
from company_bankruptcy.constants.constants import DATABASE_NAME, COLLECTION_NAME, MONGODB_COLLECTION_STR

import sys


class MongoOps:

    def __init__(self, client_url:str, database_name:str=None, collection_name:str=None):
        self.client_url = client_url
        self.database_name = database_name
        self.collection_name = collection_name

    def create_client(self):
        logging.info('Initiating MongoClient')
        client = pymongo.MongoClient(self.client_url)
        logging.info('MongoClient initiated')
        return client

    def create_database(self):
        logging.info('Creating Mongo database')
        client = self.create_client()
        database = client[self.database_name]
        logging.info(f'Mongo database {self.database_name} created')
        return database

    def create_collection(self):
        logging.info('Creating Mongo collection')
        database = self.create_database()
        collection = database[self.collection_name]
        logging.info(f'Mongo collection {self.collection_name} created')
        return collection
    
    def get_database(self, db_name:str):
        logging.info(f'Accessing {db_name} database')
        client = self.create_client()
        database = client[db_name]
        logging.info(f'{db_name} database accessed')
        return database

    def get_collection(self, coll_name:str, db_name:str):
        logging.info(f'Accessing {coll_name} collection')
        database = self.get_database(db_name)
        collection = database[coll_name]
        logging.info(f'{coll_name} collection accessed')
        return collection

    def insert_record(self, record:dict, coll_name:str, db_name:str):
        collection = self.get_collection(coll_name, db_name)
        logging.info(f'Starting record insertion into {coll_name} collection of {db_name} database')
        if isinstance(record, list):
            for data in record:
                if type(data) != dict:
                    logging.info("Records' list should have elements as dict")
                    raise TypeError("Records' list should have elements as dict")
            collection.insert_many(record)
        elif isinstance(record, dict):
            collection.insert_one(record)
        logging.info(f'Insertion into {coll_name} collection of {db_name} database completed')

    def insert_from_file(self, datafile:str, coll_name:str, db_name:str):
        logging.info(f'Starting record insertion into {coll_name} collection of {db_name} database from {datafile}')
        self.path = datafile

        if self.path.endswith('.csv'):
            df = pd.read_csv(self.path, encoding='utf-8')
        elif self.path.endswith('.xlsx'):
            df = pd.read_excel(self.path, encoding='utf-8')
        logging.info('Data is loaded as a pandas dataframe')

        logging.info('Converting the data into json')
        datajson = json.loads(df.to_json(orient='record'))
        logging.info('Conversion to json completed')

        collection = self.get_collection(coll_name, db_name)

        logging.info('Inserting json data')
        collection.insert_many(datajson)
        logging.info('Insertion completed')

    def get_records(self, coll_name:str, db_name:str):
        collection = self.get_collection(coll_name, db_name)
        retrieved_data = pd.DataFrame(list(collection.find()))
        try:
            retrieved_data.drop(columns='_id', inplace=True)
            logging.info('Loading the data from the database completed')
        except Exception as e:
            retrieved_data = pd.DataFrame()
            logging.info('Loading the data from the database failed')
            raise CustomException(e, sys)
        return retrieved_data
    
if __name__ == '__main__':

    mongo_instance = MongoOps(
        client_url=MONGODB_COLLECTION_STR
    )

    retrieved_data = mongo_instance.get_records(coll_name=COLLECTION_NAME, db_name=DATABASE_NAME)