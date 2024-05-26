import pandas as pd
import numpy as np

from company_bankruptcy.logger.logger import logging
from company_bankruptcy.exception.exception import CustomException
from company_bankruptcy.utils.utils import load_object
from company_bankruptcy.components.model_trainer import ModelTrainer
from company_bankruptcy.components.data_transformation import DataTransformation

import os
import sys

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.metrics import roc_auc_score

from urllib.parse import urlparse


class ModelEvaluation:

    def __init__(self):
        
        logging.info('Model evaluation started')

    def initiate_model_evaluation(self, test_df):

        try:

            logging.info('Setting target variable')
            y_test = test_df['Bankrupt?'].to_frame()
            logging.info('Target variable set')

            logging.info('Loading the trained models')
            model_trainer_obj = ModelTrainer()
            models_main_path = model_trainer_obj.model_trainer_config.trained_models_path
            trained_models_dict = load_object(
                os.path.join(models_main_path, 'trained_models_dict.pkl')
            )
            opt_dict = load_object(
                os.path.join(models_main_path, 'opt_dict.pkl')
            )
            logging.info('Trained models loaded')

            logging.info("Loading the features' dictionary")
            data_transformation_obj = DataTransformation()
            features_selection_dict_path = data_transformation_obj.data_transformation_config.feature_selection_dict_file_path
            feature_selection_dict = load_object(features_selection_dict_path)
            logging.info("Features' selection dictionary loaded")

            test_score_dict = {}

            logging.info('Finding test score for Average Ensemble')
            y_test_pred_prob = 0
            for model_name in trained_models_dict:
                if model_name == 'best_model_name':
                    continue
                features_list = feature_selection_dict[model_name][1]['selected_shap_feats']
                temp_prob = trained_models_dict[model_name].predict_proba(test_df[features_list])[:, 1]
                y_test_pred_prob += temp_prob
            y_test_pred_prob /= (len(trained_models_dict) - 1)
            avg_ens_score = roc_auc_score(y_test, y_test_pred_prob)
            test_score_dict['AverageEnsemble'] = avg_ens_score
            logging.info('Average Ensemble score calculated')

            logging.info('Finding test score for Optimized Ensemble')
            rfm_features_list = feature_selection_dict['RandomForestClassifier'][1]['selected_shap_feats']
            xgbm_features_list = feature_selection_dict['XGBClassifier'][1]['selected_shap_feats']
            lrm_features_list = feature_selection_dict['LogisticRegression'][1]['selected_shap_feats']
            svcm_features_list = feature_selection_dict['SVC'][1]['selected_shap_feats']
                
            preds_list = []

            for idx in opt_dict:
                opt = opt_dict[idx]['opt']
                rfm = opt_dict[idx]['rfm']
                xgbm = opt_dict[idx]['xgbm']
                lrm = opt_dict[idx]['lrm']
                svcm = opt_dict[idx]['svcm']

                rfm_probs = rfm.predict_proba(test_df[rfm_features_list])[:, 1]
                xgbm_probs = xgbm.predict_proba(test_df[xgbm_features_list])[:, 1]
                lrm_probs = lrm.predict_proba(test_df[lrm_features_list])[:, 1]
                svcm_probs = svcm.predict_proba(test_df[svcm_features_list])[:, 1]

                model_preds = np.column_stack([
                    rfm_probs,
                    xgbm_probs,
                    lrm_probs,
                    svcm_probs
                ])

                preds_list.append(opt.predict(model_preds))

            y_test_pred_prob = np.mean(np.column_stack(preds_list), axis=1)
            optimized_ens_score = roc_auc_score(y_test, y_test_pred_prob)
            test_score_dict['OptimizedEnsemble'] = optimized_ens_score
            logging.info('Optimized Ensemble score calculated')

            logging.info('Finding test score for Rank Ensemble')
            rank_ensemble_list = []
            prob_list = []
            model_names_list = []

            for model_name in trained_models_dict:
                if model_name == 'best_model_name':
                    continue
                features_list = feature_selection_dict[model_name][1]['selected_shap_feats']
                model_names_list.append(model_name)
                rank_ensemble_list.append((model_name, trained_models_dict[model_name].best_score_))
                prob_list.append(trained_models_dict[model_name].predict_proba(test_df[features_list])[:, 1])

            rank_ensemble_list = sorted(rank_ensemble_list, key=lambda x: x[1])

            y_test_pred_prob = 0
            for i in range(len(rank_ensemble_list)):
                y_test_pred_prob += (i+1) * prob_list[model_names_list.index(rank_ensemble_list[i][0])]
            y_test_pred_prob /= (len(rank_ensemble_list) * (1 + len(rank_ensemble_list)) / 2)
            rank_ens_score = roc_auc_score(y_test, y_test_pred_prob)
            test_score_dict['RankEnsemble'] = rank_ens_score
            logging.info('Rank Ensemble score calculated')
            
            for model_name in trained_models_dict:
                if model_name == 'best_model_name':
                    continue
                logging.info(f'Finding test score for {model_name}')
                features_list = feature_selection_dict[model_name][1]['selected_shap_feats']
                model = trained_models_dict[model_name]
                y_test_pred_prob = model.predict_proba(test_df[features_list])[:, 1]
                temp_score = roc_auc_score(y_test, y_test_pred_prob)
                test_score_dict[model_name] = temp_score
                logging.info(f'{model_name} score calculated')
            
            logging.info('Getting mlflow tracking uri type')
            tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info('Tracking uri got')

            logging.info('Starting mlflow')
            with mlflow.start_run():
                for model_name in test_score_dict:
                    mlflow.log_metric(f'{model_name} ROC-AUC', test_score_dict[model_name])
                    if model_name in trained_models_dict.keys():
                        model = trained_models_dict[model_name]
                        if tracking_uri_type_store != 'file':
                            # if model_name == 'XGBClassifier':
                            #     mlflow.xgboost.log_model(model, f'{model_name}', registered_model_name=f'{model_name}_model')
                            # else:
                            mlflow.sklearn.log_model(model, f'{model_name}', registered_model_name=f'{model_name}_model')
                        else:
                            # if model_name == 'XGBClassifier':
                            #     mlflow.xgboost.log_model(model, f'{model_name}')
                            # else:
                            mlflow.sklearn.log_model(model, f'{model_name}')

            logging.info('mlflow succeeded')
                

        except Exception as e:

            logging.info('Error occured during model evaluation')
            raise CustomException(e, sys)

