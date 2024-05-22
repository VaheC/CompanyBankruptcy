import pandas as pd
import numpy as np
from company_bankruptcy.logger import logging
from company_bankruptcy.exception import CustomException

import os
import sys
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    pass

class DataIngestion:

    def __init__(self):
        pass

    def initiate_data_ingestion(self):

        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e, sys)

