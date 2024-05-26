from company_bankruptcy.components.data_ingestion import DataIngestion
from company_bankruptcy.components.data_transformation import DataTransformation
from company_bankruptcy.components.model_trainer import ModelTrainer
from company_bankruptcy.components.model_evaluation import ModelEvaluation

data_ingestion_obj = DataIngestion()
train_path, test_path = data_ingestion_obj.initiate_data_ingestion()

data_transformation_obj = DataTransformation()
train_df, test_df, cv_fold_list, numerical_features = data_transformation_obj.initiate_data_transformation(
    train_path=train_path, 
    test_path=test_path
)

model_training_obj = ModelTrainer()
model_training_obj.initiate_model_training(
    train_df=train_df, 
    test_df=test_df, 
    features_dict_path=data_transformation_obj.data_transformation_config.feature_selection_dict_file_path, 
    cv_fold_list=cv_fold_list, 
    numerical_features=numerical_features
)

model_evaluation_obj = ModelEvaluation()
model_evaluation_obj.initiate_model_evaluation(test_df)
