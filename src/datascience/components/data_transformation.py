import os 
from src.datascience import logger 
from sklearn.model_selection import train_test_split
import pandas as pd 
from src.datascience.entity.config_entity import DataTransformationconfig  


class DataTransformation: 
    def __init__(self, config: DataTransformationconfig):
        self.config = config 
        
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path) 

        logger.info("Loaded data for transformation")
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")
        logger.info(f"Dtypes:\n{data.dtypes}")

        train, test = train_test_split(data, test_size=0.25, random_state=42)
        
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False) 
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False) 
        
        logger.info("Splitted the data into train and test")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}") 
        
        print("Train:", train.shape)
        print("Test:", test.shape)
