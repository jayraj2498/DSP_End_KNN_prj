import os 
from src.datascience import logger 
from sklearn.model_selection import train_test_split
import pandas as pd 
from src.datascience.entity.config_entity import DataTransformationconfig  


# here you can add diff data transformation techniques such as scaler , pca ,fe , and all etc 

class DataTransformation : 
    def __init__(self,config:DataTransformationconfig) :
        self.config = config 
        
        
    def train_test_splitting(self) :
        data= pd.read_csv(self.config.data_path) 
        
        train , test = train_test_split(data,test_size=0.25)
        
        # now we are seprating two train and test data 
        train.to_csv(os.path.join(self.config.root_dir ,"train.csv") , index=False) 
        test.to_csv(os.path.join(self.config.root_dir , "test.csv") , index=False) 
        
        logger.info("splited the data into train and test ")
        logger.info(train.shape)
        logger.info(test.shape) 
        
        
        print(train.shape)
        print(test.shape)
                     
                   
        
        