import pandas as pd
import numpy as np
from company_bankruptcy.logger import logging
from company_bankruptcy.exception import CustomException

import os
import sys
from pathlib import Path
from dataclasses import dataclass

from company_bankruptcy.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    pass

class DataTransformation:

    def __init__(self):
        pass

    def initiate_data_ingestion(self):

        try:
            pass
        except Exception as e:
            logging.info()
            raise CustomException(e, sys)

