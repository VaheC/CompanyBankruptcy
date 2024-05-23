import os
import sys
import pickle
import numpy as np
import pandas as pd

from company_bankruptcy.logger.logger import logging
from company_bankruptcy.exception.exception import CustomException

from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import r_regression, SelectKBest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier

from scipy import stats
from scipy.special import softmax
from scipy.optimize import fmin

from functools import partial

from statsmodels.stats.outliers_influence import variance_inflation_factor

from boruta import BorutaPy

import shap

from collections import Counter

from tqdm.auto import tqdm
import gc

import warnings
warnings.filterwarnings('ignore')


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            # train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e, sys)


def get_shap_features(shap_values, features, topk=10):
    '''
    Returns topk features selected using shap values

    Args:
        shap_values (object): shap explainer
        features (list): list of features' name

    Returns:
        list: topk features derived from shap values
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp,
                                fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(
        feature_importances.items(), key=lambda item: item[1], reverse=True)}
    feature_importances_norm = {k: v for k, v in sorted(
        feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
    # Prints the feature importances
    selected_topk_feats = []

    for idx, (k, v) in enumerate(feature_importances.items()):
        # print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")
        if idx <= topk:
            selected_topk_feats.append(k)

    return selected_topk_feats


class FSelector():
    '''
    Helps to select features based on BorutaPy, RFE, and various statistics
    '''

    def __init__(self, X, y, num_feats, ordinal_feats, nominal_feats, model, is_target_cat=True, select_n_feats=15):
        '''
        Initializes some parameters

        Args:
            X (pd.DataFrame): contains features' values
            y (pd.DataFrame): contains target values
            num_feats (list): list of numerical features' names
            ordinal_feats (list): list of ordinal features' names
            nominal_feats (list): list of nominal features' names
            model (model object): can be any type of model like RandomForest, LogisticRegression, etc.
            is_target_cat (bool): indicates whether the target is categorical or not
            select_n_feats (int): specifies the number of features to output
        '''

        self.X = X
        self.y = y
        self.num_feats = num_feats
        self.ordinal_feats = ordinal_feats
        self.nominal_feats = nominal_feats
        self.model = model
        self.is_target_cat = is_target_cat
        self.select_n_feats = select_n_feats

    def calculate_vif(self, X):

        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF"] = [variance_inflation_factor(
            X.values, i) for i in range(X.shape[1])]

        return vif

    def select_feats_via_vif(self):

        num_features = self.num_feats.copy()

        vif_df = self.calculate_vif(self.X[num_features])

        while vif_df[vif_df['VIF'] >= 10].shape[0] != 0:
            vif_df.sort_values('VIF', ascending=False, inplace=True)
            vif_df.reset_index(drop=True, inplace=True)
            # print(vif_df)
            elimination_candidate = vif_df.iloc[0]['features']
            # print(elimination_candidate)
            num_features = [i for i in num_features if i !=
                            elimination_candidate]
            new_X = self.X[num_features]
            vif_df = self.calculate_vif(new_X)

        return list(vif_df['features'].values)

    def get_spearmanr(self, X, y):
        # return np.array([stats.spearmanr(X.values[:, f], y.values).correlation for f in range(X.shape[1])])
        spearman_values = [stats.spearmanr(
            X.values[:, f], y.values).correlation for f in range(X.shape[1])]
        temp_sp_df = pd.DataFrame(
            {'spearman': spearman_values, 'feats': list(X.columns)})
        temp_sp_df['abs_spearman'] = np.abs(temp_sp_df['spearman'])
        temp_sp_df.sort_values('abs_spearman', ascending=False, inplace=True)
        temp_sp_df.reset_index(drop=True, inplace=True)
        return temp_sp_df.iloc[:15]['feats'].to_list()

    def get_kendalltau(self, X, y):
        # return [stats.kendalltau(X.values[:, f], y.values).correlation for f in range(X.shape[1])]
        kendall_values = [stats.spearmanr(
            X.values[:, f], y.values).correlation for f in range(X.shape[1])]
        temp_ken_df = pd.DataFrame(
            {'kendall': kendall_values, 'feats': list(X.columns)})
        temp_ken_df['abs_kendall'] = np.abs(temp_ken_df['kendall'])
        temp_ken_df.sort_values('abs_kendall', ascending=False, inplace=True)
        temp_ken_df.reset_index(drop=True, inplace=True)
        return temp_ken_df.iloc[:15]['feats'].to_list()

    def get_pointbiserialr(self, X, y):
        return [stats.pointbiserialr(X.values[:, f], y.values).correlation for f in range(X.shape[1])]

    def get_boruta_feats(self):
        feat_selector = BorutaPy(
            self.model, n_estimators='auto', verbose=2, random_state=1)
        feat_selector.fit(np.array(self.X), np.array(self.y))
        boruta_selected_features = list(
            self.X.iloc[:, feat_selector.support_].columns)
        return boruta_selected_features

    def get_kbest(self, X, feats_list, metric):
        selector = SelectKBest(metric, k=self.select_n_feats)
        selector.fit_transform(X[feats_list], self.y)
        selected_feats_idxs_list = list(selector.get_support(indices=True))
        column_names = [feats_list[i] for i in selected_feats_idxs_list]
        return column_names

    def get_rfe_feats(self):
        model_rfe = RFE(self.model, n_features_to_select=self.select_n_feats)
        model_rfe.fit(self.X, self.y)
        model_rfe_feats = list(
            self.X.iloc[:, list(model_rfe.support_)].columns)
        return model_rfe_feats

    # def get_shap_feats(self, feats_list, topk=10):
    #     model = self.model
    #     X = self.X[feats_list]
    #     model.fit(self.X, self.y)
    #     explainer = shap.Explainer(model.predict, X, max_evals = int(2 * X.shape[1] + 1), verbose=0)
    #     shap_values = explainer(X)
    #     selected_shap_features = get_feature_importances_shap_values(
    #         shap_values, features=list(X.columns), topk=topk
    #     )
    #     return selected_shap_features

    def get_features(self):

        if self.num_feats is not None:

            if self.is_target_cat:

                temp_n_feats = self.select_n_feats
                if len(self.num_feats) < self.select_n_feats:
                    self.select_n_feats = 'all'

                # self.num_kendalltau_feats = self.get_kendalltau(self.X[self.num_feats], self.y)
                self.num_f_feats = self.get_kbest(
                    X=self.X, feats_list=self.num_feats, metric=f_classif)
                self.num_mi_feats = self.get_kbest(
                    X=self.X, feats_list=self.num_feats, metric=mutual_info_classif)

                self.select_n_feats = temp_n_feats

                self.selected_num_feats = []
                # self.selected_num_feats.extend(self.num_kendalltau_feats)
                self.selected_num_feats.extend(self.num_f_feats)
                self.selected_num_feats.extend(self.num_mi_feats)

            else:

                self.vif_feats = self.select_feats_via_vif()

                temp_n_feats = self.select_n_feats
                if len(self.num_feats) < self.select_n_feats:
                    self.select_n_feats = 'all'

                self.pearson_feats = self.get_kbest(
                    X=self.X, feats_list=self.num_feats, metric=r_regression, k=self.select_n_feats)

                self.select_n_feats = temp_n_feats
                # self.num_spearmanr_feats = self.get_kbest(X=self.X, feats_list=self.num_feats, metric=stats.spearmanr, k=self.select_n_feats)
                # self.num_kendalltau_feats = self.get_kbest(X=self.X, feats_list=self.num_feats, metric=stats.kendalltau, k=self.select_n_feats)
                self.num_spearmanr_feats = self.get_spearmanr(
                    self.X[self.num_feats], self.y)
                self.num_kendalltau_feats = self.get_kendalltau(
                    self.X[self.num_feats], self.y)
                # self.num_spearmanr_feats = SelectKBest(self.get_spearmanr, k=self.select_n_feats).fit_transform(self.X[self.num_feats], self.y)
                # self.num_kendalltau_feats = SelectKBest(self.get_kendalltau, k=self.select_n_feats).fit_transform(self.X[self.num_feats], self.y)

                self.selected_num_feats = []
                self.selected_num_feats.extend(self.pearson_feats)
                self.selected_num_feats.extend(self.num_spearmanr_feats)
                self.selected_num_feats.extend(self.num_kendalltau_feats)
                # self.selected_num_feats = list(set(self.selected_num_feats))

        else:

            self.selected_num_feats = []

        if self.ordinal_feats is not None:

            if self.is_target_cat:

                temp_n_feats = self.select_n_feats
                if len(self.ordinal_feats) < self.select_n_feats:
                    self.select_n_feats = 'all'

                self.ordinal_mi_feats = self.get_kbest(
                    X=self.X, feats_list=self.ordinal_feats, metric=mutual_info_classif)
                self.ordinal_chi2_feats = self.get_kbest(
                    X=self.X, feats_list=self.ordinal_feats, metric=chi2)

                self.selected_ordinal_feats = []
                self.selected_ordinal_feats.extend(self.ordinal_mi_feats)
                self.selected_ordinal_feats.extend(self.ordinal_chi2_feats)

                self.select_n_feats = temp_n_feats

            else:

                self.ordinal_spearmanr_feats = self.get_spearmanr(
                    self.X[self.ordinal_feats], self.y)
                self.ordinal_kendalltau_feats = self.get_kendalltau(
                    self.X[self.ordinal_feats], self.y)

                # self.ordinal_spearmanr_feats = self.get_kbest(X=self.X, feats_list=self.ordinal_feats, metric=stats.spearmanr, k=self.select_n_feats)
                # self.ordinal_kendalltau_feats = self.get_kbest(X=self.X, feats_list=self.ordinal_feats, metric=stats.kendalltau, k=self.select_n_feats)

                # self.ordinal_spearmanr_feats = SelectKBest(self.get_spearmanr, k=self.select_n_feats).fit_transform(self.X[self.ordinal_feats], self.y)
                # self.ordinal_kendalltau_feats = SelectKBest(self.get_kendalltau, k=self.select_n_feats).fit_transform(self.X[self.ordinal_feats], self.y)

                self.selected_ordinal_feats = []
                self.selected_ordinal_feats.extend(
                    self.ordinal_spearmanr_feats)
                self.selected_ordinal_feats.extend(
                    self.ordinal_kendalltau_feats)
                # self.selected_ordinal_feats = list(set(self.selected_ordinal_feats))

        else:
            self.selected_ordinal_feats = []

        if self.nominal_feats is not None:

            if self.is_target_cat:

                temp_n_feats = self.select_n_feats
                if len(self.nominal_feats) < self.select_n_feats:
                    self.select_n_feats = 'all'

                self.nominal_mi_feats = self.get_kbest(
                    X=self.X, feats_list=self.nominal_feats, metric=mutual_info_classif)
                self.nominal_chi2_feats = self.get_kbest(
                    X=self.X, feats_list=self.nominal_feats, metric=chi2)

                self.selected_nominal_feats = []
                self.selected_nominal_feats.extend(self.nominal_mi_feats)
                self.selected_nominal_feats.extend(self.nominal_chi2_feats)

                self.select_n_feats = temp_n_feats

            else:

                temp_n_feats = self.select_n_feats
                if len(self.nominal_feats) < self.select_n_feats:
                    self.select_n_feats = 'all'

                self.f_feats = self.get_kbest(
                    X=self.X, feats_list=self.nominal_feats, metric=f_classif, k=self.select_n_feats)
                self.mi_feats = self.get_kbest(
                    X=self.X, feats_list=self.nominal_feats, metric=mutual_info_regression, k=self.select_n_feats)

                self.select_n_feats = temp_n_feats

                # # self.f_feats = f_classif(self.X[self.nominal_feats], self.y)[0]
                # self.f_feats = SelectKBest(f_classif, k=self.select_n_feats).fit_transform(self.X[self.nominal_feats], self.y).columns

                # # self.mi_feats = mutual_info_regression(self.X[self.nominal_feats], self.y)
                # self.mi_feats = SelectKBest(mutual_info_regression, k=self.select_n_feats).fit_transform(self.X[self.nominal_feats], self.y).columns

                self.selected_nominal_feats = []
                self.selected_nominal_feats.extend(self.f_feats)
                self.selected_nominal_feats.extend(self.mi_feats)
                # self.selected_nominal_feats = list(set(self.selected_nominal_feats))

        else:

            self.selected_nominal_feats = []

        if self.model is not None:
            # np.int = np.int32
            # np.float = np.float64
            # np.bool = np.bool_
            if isinstance(self.model, RandomForestClassifier) or isinstance(self.model, XGBClassifier):
                self.boruta_feats = self.get_boruta_feats()
            if not isinstance(self.model, SVC):
                self.rfe_feats = self.get_rfe_feats()
        else:
            self.boruta_feats = []
            self.rfe_feats = []

        if len(self.selected_num_feats) != 0:
            if isinstance(self.model, RandomForestClassifier) or isinstance(self.model, XGBClassifier):
                self.selected_num_feats.extend(self.boruta_feats)
            if not isinstance(self.model, SVC):
                self.selected_num_feats.extend(self.rfe_feats)
            num_feats_dict = dict(Counter(self.selected_num_feats))
            self.selected_num_feats = [
                i for i in num_feats_dict if num_feats_dict[i] >= 2]

        if len(self.selected_ordinal_feats) != 0:
            if isinstance(self.model, RandomForestClassifier) or isinstance(self.model, XGBClassifier):
                self.selected_ordinal_feats.extend(self.boruta_feats)
            if not isinstance(self.model, SVC):
                self.selected_ordinal_feats.extend(self.rfe_feats)
            ordinal_feats_dict = dict(Counter(self.selected_ordinal_feats))
            self.selected_ordinal_feats = [
                i for i in ordinal_feats_dict if ordinal_feats_dict[i] >= 2]

        if len(self.selected_nominal_feats) != 0:
            if isinstance(self.model, RandomForestClassifier) or isinstance(self.model, XGBClassifier):
                self.selected_nominal_feats.extend(self.boruta_feats)
            if not isinstance(self.model, SVC):
                self.selected_nominal_feats.extend(self.rfe_feats)
            nominal_feats_dict = dict(Counter(self.selected_nominal_feats))
            self.selected_nominal_feats = [
                i for i in nominal_feats_dict if nominal_feats_dict[i] >= 2]

        self.selected_feats = []
        self.selected_feats.extend(self.selected_num_feats)
        self.selected_feats.extend(self.selected_ordinal_feats)
        self.selected_feats.extend(self.selected_nominal_feats)
        if isinstance(self.model, RandomForestClassifier) or isinstance(self.model, XGBClassifier):
            self.selected_feats.extend(self.boruta_feats)
        self.selected_feats = list(set(self.selected_feats))

        # self.selected_feats = self.get_shap_feats(self.selected_feats)

        return self.selected_feats


def create_feature_selection_dict(data, cv_fold_list, numerical_features, nominal_features):
    '''
    Returns feature selection dictionary for 4 different models

    Args:
        data (pd.DataFrame): train data 
        cv_fold_list (list): contains tuples of indeces of train and validation data for each fold
        numerical_features (list): contains the names of numerical features
        nominal_features (list): contains the names of nominal features

    Returns:
        dict: contains selected features, train and validation scores, models and scalers used
    '''

    selected_features_dict = {}

    for idx in tqdm(range(1)):

        X_train = data.iloc[cv_fold_list[idx][0]].reset_index(drop=True)
        y_train = data.iloc[cv_fold_list[idx][0]
                            ]['Bankrupt?'].to_frame().reset_index(drop=True)

        X_valid = data.iloc[cv_fold_list[idx][1]].reset_index(drop=True)
        y_valid = data.iloc[cv_fold_list[idx][1]
                            ]['Bankrupt?'].to_frame().reset_index(drop=True)

        new_numerical_features = []
        for feat in numerical_features:
            X_train[f"feat{numerical_features.index(feat)}"] = X_train[feat] * \
                X_train[' Liability-Assets Flag']
            X_valid[f"feat{numerical_features.index(feat)}"] = X_valid[feat] * \
                X_valid[' Liability-Assets Flag']
            new_numerical_features.append(
                f"feat{numerical_features.index(feat)}")

        numerical_features.extend(new_numerical_features)

        # getting categorical features
        categorical_features = nominal_features.copy()

        # getting all features
        all_features = []
        all_features.extend(categorical_features)
        all_features.extend(numerical_features)

        X_train = X_train[all_features]
        X_valid = X_valid[all_features]

        models_list = [RandomForestClassifier(), XGBClassifier(
        ), LogisticRegression(), SVC(probability=True)]
        model_names_list = ['RandomForestClassifier',
                            'XGBClassifier', 'LogisticRegression', 'SVC']

        for model_idx in tqdm(range(len(model_names_list))):

            model_name = model_names_list[model_idx]

            selected_features_dict[model_name] = {}

            # feature selection
            model = models_list[model_idx]

            if isinstance(model, LogisticRegression) or isinstance(model, SVC):

                scaler = StandardScaler()

                X_train2 = scaler.fit_transform(X_train[numerical_features])
                X_train2 = pd.DataFrame(X_train2, columns=numerical_features)
                X_train2 = pd.concat(
                    [X_train2, X_train[categorical_features]], axis=1)

                fselector = FSelector(
                    X=X_train2,
                    y=y_train,
                    num_feats=numerical_features,
                    ordinal_feats=None,
                    nominal_feats=nominal_features,
                    model=model
                )

            else:

                fselector = FSelector(
                    X=X_train,
                    y=y_train,
                    num_feats=numerical_features,
                    ordinal_feats=None,
                    nominal_feats=nominal_features,
                    model=model
                )

            selected_features = fselector.get_features()

            if len(selected_features) == 0:
                continue

            # selecting features using shap values
            if isinstance(model, LogisticRegression) or isinstance(model, SVC):

                X_valid2 = scaler.transform(X_valid[numerical_features])
                X_valid2 = pd.DataFrame(X_valid2, columns=numerical_features)
                X_valid2 = pd.concat(
                    [X_valid2, X_valid[categorical_features]], axis=1)

                X_train_filtered = X_train2[selected_features]
                X_valid_filtered = X_valid2[selected_features]

            else:

                X_train_filtered = X_train[selected_features]
                X_valid_filtered = X_valid[selected_features]

            # model training using selected features
            model.fit(X_train_filtered, y_train)

            explainer = shap.Explainer(
                model.predict,
                X_train_filtered,
                # max_evals = int(2 * X_train_filtered.shape[1] + 1),
                verbose=0
            )
            shap_values = explainer(X_train_filtered)
            selected_shap_features = get_shap_features(
                shap_values,
                features=list(X_train_filtered.columns),
                topk=10
            )

            # model training using shap features
            model = models_list[model_idx]
            model.fit(X_train_filtered[selected_shap_features], y_train)

            # metric calculation
            y_train_pred = model.predict(
                X_train_filtered[selected_shap_features])
            y_train_pred_prob = model.predict_proba(
                X_train_filtered[selected_shap_features])[:, 1]

            y_valid_pred = model.predict(
                X_valid_filtered[selected_shap_features])
            y_valid_pred_prob = model.predict_proba(
                X_valid_filtered[selected_shap_features])[:, 1]

            train_acc = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            train_roc_auc = roc_auc_score(y_train, y_train_pred_prob)

            valid_acc = accuracy_score(y_valid, y_valid_pred)
            valid_f1 = f1_score(y_valid, y_valid_pred)
            valid_roc_auc = roc_auc_score(y_valid, y_valid_pred_prob)

            selected_features_dict[model_name][idx+1] = {}
            selected_features_dict[model_name][idx +
                                               1]['selected_feats'] = selected_features
            selected_features_dict[model_name][idx +
                                               1]['selected_shap_feats'] = selected_shap_features
            selected_features_dict[model_name][idx+1]['train_acc'] = train_acc
            selected_features_dict[model_name][idx+1]['train_f1'] = train_f1
            selected_features_dict[model_name][idx +
                                               1]['train_roc_auc'] = train_roc_auc
            selected_features_dict[model_name][idx+1]['valid_acc'] = valid_acc
            selected_features_dict[model_name][idx+1]['valid_f1'] = valid_f1
            selected_features_dict[model_name][idx +
                                               1]['valid_roc_auc'] = valid_roc_auc
            selected_features_dict[model_name][idx+1]['model'] = model
            if isinstance(model, LogisticRegression) or isinstance(model, SVC):
                selected_features_dict[model_name][idx+1]['scaler'] = scaler

            # print(f"##### {model_name} #####")
            # print(f"Selected features: {selected_features}")
            # print("Train:")
            # print(f"Accuracy: {train_acc:.5f}, F1: {train_f1:.5f}, ROC-AUC: {train_roc_auc:.5f}")
            # print("Validation:")
            # print(f"Accuracy: {valid_acc:.5f}, F1: {valid_f1:.5f}, ROC-AUC: {valid_roc_auc:.5f}")

            logging.info("##### %(model_name)s #####")
            logging.info(f"Selected features: {selected_features}")
            logging.info('Train:')
            logging.info(
                f"Accuracy: {train_acc:.5f}, F1: {train_f1:.5f}, ROC-AUC: {train_roc_auc:.5f}")
            logging.info('Validation:')
            logging.info(
                f"Accuracy: {valid_acc:.5f}, F1: {valid_f1:.5f}, ROC-AUC: {valid_roc_auc:.5f}")

        del X_train, y_train, X_valid, y_valid, X_train_filtered, X_valid_filtered, model
        gc.collect()

    return selected_features_dict


def get_mean_ensemble_prediction(prob_list):
    prob_array = np.vstack(prob_list).T
    return np.mean(prob_array, axis=1)


class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        X_coef = X * coef
        preds = np.sum(X_coef, axis=1)
        auc_score = roc_auc_score(y, preds)
        return -1 * auc_score

    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        X_coef = X * self.coef_
        preds = np.sum(X_coef, axis=1)
        return preds


def get_optimized_ensemble(train_df, test_df, cv_fold_list, selected_features_dict, trained_models_dict, numerical_features):
    '''
    Finds the optimized weights for ensembling using the train data and evaluates it on test data

    Args:
        train_df (pd.DataFrame): train data 
        test_df (pd.DataFrame): test data 
        cv_fold_list (list): contains tuples of indeces of train and validation data for each fold
        selected_features_dict (dict): selected features dictionary where keys are models' names
        trained_models_dict (dict): trained models dictionary where keys are models' names
        numerical_features (list): contains the names of numerical features

    Returns:
        dict: contains all optimized weights for each fold
        float: ROC-AUC score
    '''

    opt_dict = {}

    test_preds_list = []
    # valid_preds_list = []

    X_test_rf = test_df[selected_features_dict['RandomForestClassifier']
                        [1]['selected_shap_feats']]
    X_test_xgb = test_df[selected_features_dict['XGBClassifier']
                         [1]['selected_shap_feats']]
    X_test_lr = test_df[selected_features_dict['LogisticRegression']
                        [1]['selected_shap_feats']]
    X_test_svc = test_df[selected_features_dict['SVC']
                         [1]['selected_shap_feats']]

    y_test = test_df['Bankrupt?'].to_frame()

    for idx in range(len(cv_fold_list)):

        logging.info(f'Starting calculations for Fold {idx+1}')

        X_train = train_df.iloc[cv_fold_list[idx][0]].reset_index(drop=True)
        y_train = train_df.iloc[cv_fold_list[idx][0]
                                ]['Bankrupt?'].to_frame().reset_index(drop=True)

        X_valid = train_df.iloc[cv_fold_list[idx][1]].reset_index(drop=True)
        y_valid = train_df.iloc[cv_fold_list[idx][1]
                                ]['Bankrupt?'].to_frame().reset_index(drop=True)

        # RandomForest
        logging.info('Starting RandomForest calculations')
        rf_selected_features = selected_features_dict['RandomForestClassifier'][1]['selected_shap_feats']
        X_train_rf = X_train[rf_selected_features]
        X_valid_rf = X_valid[rf_selected_features]

        rf_gscv = trained_models_dict['RandomForestClassifier']

        rfm = RandomForestClassifier(**rf_gscv.best_params_)
        rfm.fit(X_train_rf, y_train)
        rfm_valid_probs = rfm.predict_proba(X_valid_rf)[:, 1]

        rfm_test_probs = rfm.predict_proba(X_test_rf)[:, 1]
        logging.info('RandomForest calculations completed')

        # XGBoost
        logging.info('Starting XGBoost calculations')
        xgb_selected_features = selected_features_dict['XGBClassifier'][1]['selected_shap_feats']
        X_train_xgb = X_train[xgb_selected_features]
        X_valid_xgb = X_valid[xgb_selected_features]

        xgb_gscv = trained_models_dict['XGBClassifier']

        xgbm = XGBClassifier(**xgb_gscv.best_params_)
        xgbm.fit(X_train_xgb, y_train)
        xgbm_valid_probs = xgbm.predict_proba(X_valid_xgb)[:, 1]
        xgbm_test_probs = xgbm.predict_proba(X_test_xgb)[:, 1]
        logging.info('XGBoost calculations completed')

        # LogisticRegression
        logging.info('Starting LogisticRegression calculations')
        lr_selected_features = selected_features_dict['LogisticRegression'][1]['selected_shap_feats']
        X_train_lr = X_train[lr_selected_features]
        X_valid_lr = X_valid[lr_selected_features]

        lr_gscv = trained_models_dict['LogisticRegression']

        lr_params = {k.replace('model__', ''): v for k,
                     v in lr_gscv.best_params_.items()}
        selected_shap_features = selected_features_dict['LogisticRegression'][1]['selected_shap_feats']
        num_feat = [
            col for col in selected_shap_features if col in numerical_features]
        num_trans = Pipeline([('scale', StandardScaler())])
        preprocessor = ColumnTransformer(
            transformers=[('num', num_trans, num_feat)], remainder='passthrough')
        lrm = Pipeline(
            [
                ('preproc', preprocessor),
                ('lr', LogisticRegression(**lr_params))
            ]
        )
        lrm.fit(X_train_lr, y_train)
        lrm_valid_probs = lrm.predict_proba(X_valid_lr)[:, 1]
        lrm_test_probs = lrm.predict_proba(X_test_lr)[:, 1]
        logging.info('LogisticRegression calculations completed')

        # SVC
        logging.info('Starting SVC calculations')
        svc_selected_features = selected_features_dict['SVC'][1]['selected_shap_feats']
        X_train_svc = X_train[svc_selected_features]
        X_valid_svc = X_valid[svc_selected_features]

        svc_gscv = trained_models_dict['SVC']

        svc_params = {k.replace('model__', ''): v for k,
                      v in svc_gscv.best_params_.items()}
        selected_shap_features = selected_features_dict['SVC'][1]['selected_shap_feats']
        num_feat = [
            col for col in selected_shap_features if col in numerical_features]
        num_trans = Pipeline([('scale', StandardScaler())])
        preprocessor = ColumnTransformer(
            transformers=[('num', num_trans, num_feat)], remainder='passthrough')
        svcm = Pipeline(
            [
                ('preproc', preprocessor),
                ('svc', SVC(probability=True, **svc_params))
            ]
        )
        svcm.fit(X_train_svc, y_train)
        svcm_valid_probs = svcm.predict_proba(X_valid_svc)[:, 1]
        svcm_test_probs = svcm.predict_proba(X_test_svc)[:, 1]
        logging.info('SVC calculations completed')

        logging.info('Optimizing Ensemble weights')
        valid_preds = np.column_stack([
            rfm_valid_probs,
            xgbm_valid_probs,
            lrm_valid_probs,
            svcm_valid_probs
        ])

        opt = OptimizeAUC()
        opt.fit(valid_preds, y_valid)
        opt_dict[idx] = opt
        logging.info('Optimization finished')

        # valid_preds_list.append(opt.predict(valid_preds))

        logging.info('Calculating predictions for test set')
        test_preds = np.column_stack([
            rfm_test_probs,
            xgbm_test_probs,
            lrm_test_probs,
            svcm_test_probs
        ])

        test_preds_list.append(opt.predict(test_preds))
        logging.info('Test set predictions calculated')

    logging.info('Getting the score for test set')
    opt_y_test_pred_prob = np.mean(np.column_stack(test_preds_list), axis=1)
    opt_test_roc_auc = roc_auc_score(y_test, opt_y_test_pred_prob)
    logging.info('Test score calculated')

    return (opt_dict, opt_test_roc_auc)


def find_optimal_model(train_df, test_df, features_dict_path, cv_fold_list, numerical_features):
    '''
    Finds the best model for the train data and evaluates it on test data

    Args:
        train_df (pd.DataFrame): train data 
        test_df (pd.DataFrame): test data 
        features_dict_path (str): path to selected features dictionary
        cv_fold_list (list): contains tuples of indeces of train and validation data for each fold
        numerical_features (list): contains the names of numerical features

    Returns:
        dict: contains all trained models and the name of the best model
        dict: contains all optimized weights of ensembling for each fold
    '''
    logging.info('Loading selected features dictionary')
    selected_features_dict = load_object(file_path=features_dict_path)
    logging.info('Selected features dictionary loaded')

    models_list = [RandomForestClassifier(), XGBClassifier(),
                   LogisticRegression(), SVC(probability=True)]
    model_names_list = ['RandomForestClassifier',
                        'XGBClassifier', 'LogisticRegression', 'SVC']
    model_params_list = [
        {
            'n_estimators': [5, 10, 15, 25, 50, 100, 120, 300, 500],
            'max_depth': [2, 3, 5, 8, 15, 25, 30, None]
        },
        {
            'eta': [0.01, 0.015, 0.025, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9],
            'max_depth': [3, 5, 6, 7, 9, 12, 15, 17, 25],
            'n_estimators': [50, 100, 150, 200, 500, 1000]
        },
        {'model__penalty': ['l1', 'l2'], 'model__C': [
            0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        {'model__C': [1, 10, 100, 1000], 'model__gamma': [
            1, 0.1, 0.001, 0.0001], 'model__kernel': ['linear', 'rbf']}
    ]

    trained_models_dict = {}

    best_score = 0
    best_model_name = None

    X_train = train_df[selected_shap_features]
    y_train = train_df['Bankrupt?'].to_frame()

    X_test = test_df[selected_shap_features]
    y_test = test_df['Bankrupt?'].to_frame()

    y_train_pred_prob_list = []
    y_test_pred_prob_list = []
    rank_ensemble_list = []

    for model_idx in tqdm(range(len(model_names_list))):

        # y_train_pred_prob = np.zeros(X_train.shape)

        model_name = model_names_list[model_idx]

        selected_shap_features = selected_features_dict[model_name][1]['selected_shap_feats']

        logging.info(f'Starting {model_name} training')
        params_dict = model_params_list[model_idx]

        model = models_list[model_idx]

        if isinstance(model, LogisticRegression) or isinstance(model, SVC):
            num_feat = [
                col for col in selected_shap_features if col in numerical_features]
            num_trans = Pipeline([('scale', StandardScaler())])
            preprocessor = ColumnTransformer(
                transformers=[('num', num_trans, num_feat)], remainder='passthrough')
            pipe = Pipeline(
                [
                    ('preproc', preprocessor),
                    ('model', model)
                ]
            )

            model_gscv = GridSearchCV(
                pipe,
                param_grid=params_dict,
                scoring='roc_auc',
                cv=cv_fold_list,
                n_jobs=-1,
                verbose=4
            )
        else:
            model_gscv = GridSearchCV(
                model,
                param_grid=params_dict,
                scoring='roc_auc',
                cv=cv_fold_list,
                n_jobs=-1,
                verbose=4
            )

        model_gscv.fit(X_train, y_train)
        logging.info(f'{model_name} training finished')

        trained_models_dict[model_name] = model_gscv

        rank_ensemble_list.append((model_name, model_gscv.best_score_))

        # for train_idxs, valid_idxs in cv_fold_list:
        #     temp_model = models_list[model_idx]
        #     y_train_pred_prob[valid_idxs, :] = model_gscv.predict_proba(X_train[valid_idxs, :])[:, 1]
        # y_train_pred_prob_list.append(y_train_pred_prob)

        logging.info('Getting ROC-AUC for test set')
        y_test_pred_prob = model_gscv.predict_proba(X_test)[:, 1]
        y_test_pred_prob_list.append(y_test_pred_prob)
        test_roc_auc = roc_auc_score(y_test, y_test_pred_prob)
        logging.info(
            f'{model_name}:  Validation score = {model_gscv.best_score_:.4f}, Test score = {test_roc_auc:.4f}')

        if test_roc_auc > best_score:
            best_score = test_roc_auc
            best_model_name = model_name

    logging.info('Getting Average Ensemble score')
    # avg_ens_y_train_pred_prob = get_mean_ensemble_prediction(y_train_pred_prob_list)
    # avg_ens_train_roc_auc = roc_auc_score(y_test, avg_ens_y_train_pred_prob)

    avg_ens_y_test_pred_prob = get_mean_ensemble_prediction(
        y_test_pred_prob_list)
    avg_ens_test_roc_auc = roc_auc_score(y_test, avg_ens_y_test_pred_prob)
    logging.info(f'Average Ensemble: Test score = {avg_ens_test_roc_auc:.4f}')
    # logging.info(f'Average Ensemble:  Validation score = {avg_ens_train_roc_auc:.4f}, Test score = {avg_ens_test_roc_auc:.4f}')

    if avg_ens_test_roc_auc > best_score:
        best_score = avg_ens_test_roc_auc
        best_model_name = 'Average Ensemble'

    logging.info('Getting Rank Ensemble score')
    rank_ensemble_list = sorted(rank_ensemble_list, key=lambda x: x[1])

    # rank_ens_y_train_pred_prob = 0
    rank_ens_y_test_pred_prob = 0
    for i in range(len(rank_ensemble_list)):
        # rank_ens_y_train_pred_prob += (i+1) * y_train_pred_prob_list[model_names_list.index(rank_ensemble_list[i][0])]
        rank_ens_y_test_pred_prob += (
            i+1) * y_test_pred_prob_list[model_names_list.index(rank_ensemble_list[i][0])]
    # rank_ens_y_train_pred_prob /= len(rank_ensemble_list) * (1+ len(rank_ensemble_list)) / 2
    rank_ens_y_test_pred_prob /= len(rank_ensemble_list) * \
        (1 + len(rank_ensemble_list)) / 2
    rank_ens_test_roc_auc = roc_auc_score(y_test, rank_ens_y_test_pred_prob)

    logging.info(f'Rank Ensemble:  Test score = {rank_ens_test_roc_auc:.4f}')
    # logging.info(f'Rank Ensemble:  Validation score = {rank_ens_y_train_pred_prob:.4f}, Test score = {rank_ens_y_test_pred_prob:.4f}')

    if rank_ens_test_roc_auc > best_score:
        best_score = rank_ens_test_roc_auc
        best_model_name = 'Rank Ensemble'

    logging.info('Getting Optimized Ensemble score')
    opt_dict, opt_test_roc_auc = get_optimized_ensemble(
        train_df,
        test_df,
        cv_fold_list,
        selected_features_dict,
        trained_models_dict,
        numerical_features
    )

    logging.info(f'Optimized Ensemble:  Test score = {opt_test_roc_auc:.4f}')

    if opt_test_roc_auc > best_score:
        best_score = opt_test_roc_auc
        best_model_name = 'Optimized Ensemble'

    trained_models_dict['best_model_name'] = best_model_name

    logging.info(f'{best_model_name} is the best model')

    return (trained_models_dict, opt_dict)
