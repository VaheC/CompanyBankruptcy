import streamlit as st 
import pandas as pd
import numpy as np

import os
import sys

from company_bankruptcy.components.model_trainer import ModelTrainer
from company_bankruptcy.components.data_transformation import DataTransformation
from company_bankruptcy.utils.utils import load_object
from company_bankruptcy.logger.logger import logging
from company_bankruptcy.exception.exception import CustomException

st.set_page_config(
    page_title='Default Predictor',
    layout='centered'
)

try:

    st.title('Company Default Predictor')
    
    logging.info('Initiating dictionaries')
    if 'trained_models_dict' not in st.session_state:
        model_trainer_obj = ModelTrainer()
        trained_models_dict = load_object(
            os.path.join(
                model_trainer_obj.model_trainer_config.trained_models_path,
                'trained_models_dict.pkl'
            )
        )
        opt_dict = load_object(
            os.path.join(
                model_trainer_obj.model_trainer_config.trained_models_path,
                'opt_dict.pkl'
            )
        )

        data_transformation_obj = DataTransformation()
        feature_selection_dict = load_object(
            data_transformation_obj.data_transformation_config.feature_selection_dict_file_path
        )

        st.session_state['trained_models_dict'] = trained_models_dict
        st.session_state['opt_dict'] = opt_dict
        st.session_state['feature_selection_dict'] = feature_selection_dict

    else:

        trained_models_dict = st.session_state['trained_models_dict']
        opt_dict = st.session_state['opt_dict']
        feature_selection_dict = st.session_state['feature_selection_dict']
    logging.info('Dictionaries initiated')    
    
    logging.info('Checking button clicked')
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    logging.info(f'Button check passed with value {st.session_state.clicked}')
    

    st.subheader('Please, fill in the input boxes and click on submit to get the default probability.')

    best_model_name = trained_models_dict['best_model_name']
    
    logging.info("Getting features' list")
    if best_model_name in ['Average Ensemble', 'Optimized Ensemble', 'Rank Ensemble']:
        features_list = []
        for model_name in feature_selection_dict:
            features_list.extend(
                feature_selection_dict[model_name][1]['selected_shap_feats']
            )
        features_list = list(set(features_list))
    else:
        features_list = feature_selection_dict[best_model_name][1]['selected_shap_feats']
    logging.info("Features' list found")

    n_cols = 2
    n_rows = int((len(features_list) - len(features_list) % n_cols) / n_cols)
    if len(features_list) % n_cols != 0:
        n_rows += 1

    logging.info('Constructing the app input structure')
    input_dict = {}
    feature_idx = 0
    for i in range(n_rows):

        temp_input_container = st.container()

        with temp_input_container:
            col1, col2 = st.columns(n_cols)
            if i <= n_rows - 1 and len(features_list) % 2 == 0:
                input_dict[features_list[feature_idx]] = [
                    col1.number_input(
                        features_list[feature_idx],
                        format='%.6f'
                    )
                ]
                input_dict[features_list[feature_idx+1]] = [
                    col2.number_input(
                        features_list[feature_idx+1],
                        format='%.6f'
                    )
                ]
            else:
                input_dict[features_list[feature_idx]] = [
                    col1.number_input(
                        features_list[feature_idx],
                        format='%.6f'
                    )
                ]

        feature_idx += 2

    logging.info('Input structure constructed')

    def set_button_click():
        st.session_state.clicked = True

    st.button('Submit', on_click=set_button_click)

    if st.session_state.clicked:

        logging.info(f'Calculating prob for {best_model_name}')

        input_df = pd.DataFrame(input_dict)

        logging.info(f'input_df: {input_df}')

        if best_model_name == 'Average Ensemble':
            
            default_prob = 0
            for model_name in trained_models_dict:
                if model_name == 'best_model_name':
                    continue
                temp_features_list = feature_selection_dict[model_name][1]['selected_shap_feats']
                temp_prob = trained_models_dict[model_name].predict_proba(input_df[temp_features_list])[:, 1]
                default_prob += temp_prob
            default_prob /= (len(trained_models_dict) - 1)

        elif best_model_name == 'Optimized Ensemble':

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

                rfm_probs = rfm.predict_proba(input_df[rfm_features_list])[:, 1]
                xgbm_probs = xgbm.predict_proba(input_df[xgbm_features_list])[:, 1]
                lrm_probs = lrm.predict_proba(input_df[lrm_features_list])[:, 1]
                svcm_probs = svcm.predict_proba(input_df[svcm_features_list])[:, 1]

                model_preds = np.column_stack([
                    rfm_probs,
                    xgbm_probs,
                    lrm_probs,
                    svcm_probs
                ])

                preds_list.append(opt.predict(model_preds))

            default_prob = np.mean(np.column_stack(preds_list), axis=1)

        elif best_model_name == 'Rank Ensemble':

            rank_ensemble_list = []
            prob_list = []
            model_names_list = []

            for model_name in trained_models_dict:
                if model_name == 'best_model_name':
                    continue
                temp_features_list = feature_selection_dict[model_name][1]['selected_shap_feats']
                model_names_list.append(model_name)
                rank_ensemble_list.append((model_name, trained_models_dict[model_name].best_score_))
                prob_list.append(trained_models_dict[model_name].predict_proba(input_df[temp_features_list])[:, 1])

            rank_ensemble_list = sorted(rank_ensemble_list, key=lambda x: x[1])

            default_prob = 0
            for i in range(len(rank_ensemble_list)):
                default_prob += (i+1) * prob_list[model_names_list.index(rank_ensemble_list[i][0])]
            default_prob /= (len(rank_ensemble_list) * (1 + len(rank_ensemble_list)) / 2)

        else:
            model = trained_models_dict[best_model_name]
            temp_features_list = feature_selection_dict[best_model_name][1]['selected_shap_feats']
            default_prob = model.predict_proba(input_df[temp_features_list])[:, 1]

        st.write(f"Default probability: {default_prob[0]:.4f}")

        logging.info(f'Default prob: {default_prob[0]:.4f}')

except Exception as e:
    logging.info('Error occured while creating streamlit app')
    raise CustomException(e, sys)

