import os
os.environ["NUMEXPR_MAX_THREADS"] = "12"
import pandas as pd
from src.datascience import logger 
from sklearn.linear_model import ElasticNet 
import joblib  
from src.datascience.entity.config_entity import ModelTrainerConfig



# this particular function will train our entire ml model 

class ModelTrainer : 
    def __init__(self,config:ModelTrainerConfig) :
        self.config = config 
        
        
    def train(self) :
        train_data = pd.read_csv(self.config.train_data_path )  
        test_data = pd.read_csv(self.config.test_data_path) 
        
        
        train_x = train_data.drop(columns=[self.config.target_column],axis=1)
        train_y = train_data[self.config.target_column]
        test_x = test_data.drop(columns=[self.config.target_column],axis=1)
        test_y = test_data[self.config.target_column]
        
        
        lr = ElasticNet(alpha=self.config.alpha , l1_ratio=self.config.l1_ratio , random_state=42)
        lr.fit(train_x,train_y) 
        
        
        joblib.dump(lr ,os.path.join(self.config.root_dir ,self.config.model_name))
        
         