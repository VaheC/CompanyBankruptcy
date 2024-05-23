import pandas as pd
import numpy as np
from company_bankruptcy.logger.logger import logging
from company_bankruptcy.exception.exception import CustomException

import os
import sys
from pathlib import Path
from dataclasses import dataclass

from company_bankruptcy.utils.utils import save_object, evaluate_model, load_object, find_optimal_model


@dataclass
class ModelTrainerConfig:
    trained_models_path = os.path.join('artifacts', 'models')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_df, test_df, features_dict_path, cv_fold_list, numerical_features):

        try:

            logging.info('Creating a directory to save trained models')
            os.makedirs(
                self.model_trainer_config.trained_models_path, exist_ok=True)
            logging.info("Models' directory created")

            logging.info('Finding the best model')
            trained_models_dict, opt_dict = find_optimal_model(
                train_df,
                test_df,
                features_dict_path,
                cv_fold_list,
                numerical_features
            )

            logging.info(
                "Saving trained models' and ensemble optimized weights' dictionaries")
            save_object(
                file_path=os.path.join(
                    self.model_trainer_config.trained_models_path, 'trained_models_dict.pkl'),
                obj=trained_models_dict
            )

            save_object(
                file_path=os.path.join(
                    self.model_trainer_config.trained_models_path, 'opt_dict.pkl'),
                obj=opt_dict
            )
            logging.info('Saving completed')

        except Exception as e:
            logging.info('Error occured during model training')
            raise CustomException(e, sys)
