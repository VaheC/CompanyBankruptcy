import pandas as pd
from company_bankruptcy.logger.logger import logging
from company_bankruptcy.exception.exception import CustomException

import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold

from company_bankruptcy.utils.utils import save_object, create_feature_selection_dict

@dataclass
class DataTransformationConfig:
    feature_selection_dict_file_path = os.path.join('artifacts', 'feature_selection_dict.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path, n_cv_folds=10):

        try:
            logging.info('Loading training data')
            train_df = pd.read_csv(train_path)
            logging.info('Training data loaded')

            logging.info('Loading testing data')
            test_df = pd.read_csv(test_path)
            logging.info('Testing data loaded')

            logging.info('Removing Net Income Flag')
            train_df.drop(columns=' Net Income Flag', inplace=True)
            test_df.drop(columns=' Net Income Flag', inplace=True)
            logging.info('Net Income Flag removed')

            logging.info('Specifying nominal and numerical features as list')
            nominal_features = [' Liability-Assets Flag']
            numerical_features = [col for col in train_df.columns if col not in nominal_features and col!='Bankrupt?']
            logging.info('Nominal and numerical features specified')

            logging.info(f'Creating {n_cv_folds} CV folds for train data')
            skfold = StratifiedKFold(n_splits=n_cv_folds, random_state=42, shuffle=True)
            skfold_list = []
            for train_idxs, valid_idxs in skfold.split(train_df, y=train_df['Bankrupt?']):
                skfold_list.append((train_idxs, valid_idxs))
            logging.info('CV folds created')
            
            logging.info('Creating new columns using categorical and numerical iteractions')
            for feat in numerical_features:
                train_df[f"feat{numerical_features.index(feat)}"] = train_df[feat] * train_df[' Liability-Assets Flag']
                test_df[f"feat{numerical_features.index(feat)}"] = test_df[feat] * test_df[' Liability-Assets Flag']
            logging.info('New columns created')

            logging.info('Starting feature selection')
            selected_features_dict = create_feature_selection_dict(
                data=train_df, 
                cv_fold_list=skfold_list, 
                numerical_features=numerical_features, 
                nominal_features=nominal_features
            )
            logging.info('Feature selection completed')

            logging.info('Saving feature selection dictionary as pkl file')
            save_object(
                file_path=self.data_transformation_config.feature_selection_dict_file_path,
                obj=selected_features_dict
            )
            logging.info('Dictionary saved')

            return (train_df, test_df)

        except Exception as e:
            logging.info('Error occured during data transformation')
            raise CustomException(e, sys)

if __name__ == '__main__':
    
    data_transformation_obj = DataTransformation()
    train_df, test_df = data_transformation_obj.initiate_data_transformation(
        train_path='artifacts\\train_data.csv', 
        test_path='artifacts\\test_data.csv'
    )