import os 
import pandas as pd 
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score 
import mlflow
import numpy as np 
import joblib 
import mlflow.sklearn 
from urllib.parse import urlparse  

from src.datascience.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
from src.datascience.constants import * 
from src.datascience.utils.common import read_yaml , create_directories , save_json 


# os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/jayraj2498/DSP_End_KNN_prj.mlflow"
# os.environ['MLFLOW_TRACKING_USERNAME'] = "jayraj2498"
# os.environ['MLFLOW_TRACKING_PASSWORD'] = "777e2be0b0c43fcc2efbc898716cbaebe35c912b"





# model evaluation component 
# it is the code to save model locally cuase mlflow model registry is not supported in dagshub

class ModelEvaluation : 
    def __init__(self, config:ModelEvaluationConfig) :
        self.config= config 
        
        
    def eval_metrics(self,actual,pred): 
        rmse = np.sqrt(mean_squared_error(actual,pred)) 
        mae = mean_absolute_error(actual,pred) 
        r2 = r2_score(actual,pred) 
        
        return rmse,mae,r2 
    
    def log_into_mlflow(self) :
        
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)  
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column] 
        
        mlflow.set_registry_uri(self.config.mlflow_uri) 
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme 
        


        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

            # save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # log params
            mlflow.log_params(self.config.all_params)

            # log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            #  disable log_model (unsupported on DagsHub MLflow)
            # mlflow.sklearn.log_model(model, "model")

            #  instead, save model locally
            model_path = Path("artifacts/model_evaluation/saved_model.pkl")
            joblib.dump(model, model_path)
            print(f"Model saved locally at {model_path}") 



# <---------------------------------><---------------------------------------------------------><--------> 

# # # model evaluation component 

# class ModelEvaluation : 
#     def __init__(self, config:ModelEvaluationConfig) :
#         self.config= config 
        
        
#     def eval_metrics(self,actual,pred): 
#         rmse = np.sqrt(mean_squared_error(actual,pred)) 
#         mae = mean_absolute_error(actual,pred) 
#         r2 = r2_score(actual,pred) 
        
#         return rmse,mae,r2 
    
#     def log_into_mlflow(self) :
        
#         test_data = pd.read_csv(self.config.test_data_path)
#         model = joblib.load(self.config.model_path)  
        
#         test_x = test_data.drop([self.config.target_column], axis=1)
#         test_y = test_data[self.config.target_column] 
        
#         mlflow.set_registry_uri(self.config.mlflow_uri) 
#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme 
        
        
#         with mlflow.start_run():
            
#             predicted_qualities = model.predict(test_x)

#             (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

#             # saving metrics locally
#             scores = {"rmse": rmse, "mae": mae, "r2": r2}
#             save_json(path=Path(self.config.metric_file_name), data=scores)

#             # log parameters
#             mlflow.log_params(self.config.all_params)

#             # log metrics (fixed: use log_metric, not log_metrics for single values)
#             mlflow.log_metric("rmse", rmse)
#             mlflow.log_metric("r2", r2)
#             mlflow.log_metric("mae", mae)

#             # model registry does not work with file store
#             if tracking_url_type_store != "file":
#                 mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
#             else:
#                 mlflow.sklearn.log_model(model, "model")

        
        



