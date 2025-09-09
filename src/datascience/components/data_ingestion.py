 
import os 
from urllib import request
from src.datascience import logger
import zipfile
from src.datascience.entity.config_entity import (DataIngestionConfig)


# Update the component for data ingestion 
import requests

class DataIngestion :
    def __init__(self,config:DataIngestionConfig) :
        self.config = config 
        
    # this function is for downloading the zip file     
    def download_file(self) :
        if not os.path.exists(self.config.local_data_file) :
            filename , headers = request.urlretrieve(
                url = self.config.source_URL ,
                filename = self.config.local_data_file
            ) 
            
            logger.info(f"{filename} download with following info : {headers}") 
        else :
            logger.info(f"file already exists")    
            
    #  unzip the file        
    def extract_zip_file(self) :
        '''
        zip_file_path : str
        Extract the zip file into the data directory 
        Fuction return none 
        '''
        
        unzip_path = self.config.unzip_dir 
        os.makedirs(unzip_path , exist_ok=True) 
        with zipfile.ZipFile(self.config.local_data_file ,'r') as zip_ref :
            zip_ref.extractall(unzip_path)
    
    
    