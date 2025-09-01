import os 
import yaml
from src.datascience import logger 
import json 
import joblib 
from ensure import ensure_annotations 
from box import ConfigBox
from pathlib import Path
from typing import Any  
from box.exceptions import BoxValueError , BoxKeyError






# this fucntion will read yaml file 
@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox :
    """ read yaml file and returns 
    Args:
        path_to_yaml(str) :path like input
        
    Raises :
        ValueError : if yaml file is empty
        e : empty file
        
    Returns:
        ConfigBox : ConfigBox     
        
    """ 
    try :
        with open (path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded successfully") 
            return ConfigBox(content) 
    except BoxValueError :
        raise ValueError("yaml file is empty") 
    except Exception as e :
        raise e 
    
    
    
    
# this function is for creating the directories    
@ensure_annotations
def create_directories(path_to_directories:list , verbose=True) :
    """
    Create list of directories 
    
    Args : 
        Path_to_directories(list) : list of path of directories 
        ignore_log (bool , optional) : ignore if multiple directories are to be created. Defaults to False. 
    """
    
    for path in path_to_directories :
        os.makedirs(path , exist_ok=True) 
        if verbose :
            logger.info(f"created directory at : {path}")
    
    
    
    
# funt for saving the json file     
@ensure_annotations 
def save_json(path:Path , data:dict) :
    """ 
    Save json 
    
    Args:
    Path(path) :path of json file 
    data(dict) : data to be saved  
    """
    with open(Path ,"w") as f :
        json.dump(data , f , indent=4) 
        
    logger.info(f"json file saved at : {path}") 
    
    
    
    
# for loading  the json file 
@ensure_annotations 
def load_json(path:Path) -> ConfigBox :
    """ 
    load json file data 
    Args:
        path(Path) : path of json file 
        
    Returns:
        ConfigBox : data as class atribute insted of dict 
    """
    
    with open(Path) as f :
        content = json.load(f) 
        
    logger.info(f"json file loaded successfully from : {Path}") 
    return ConfigBox(content)
        
        
        
        
# This function is for Saving the model 
@ensure_annotations 
def save_bin(data:Any ,path:Path) :
    """ 
    Save binary file data 
    
    Args: 
        data(Any) : data to be saved as binary
        path(Path) : path of binary file
    
    """
    joblib.dump(value=data ,filename=path)
    logger.info(f"binary file saved at : {path}")



# this function is for loading the model 
@ensure_annotations
def load_bin(path:Path)-> Any : 
    """ 
    load binary file data
    
    Args:
    path(Path) : path to binary file 
    
    Returns:
        Any : object stored in the file
    """
    
    data = joblib.load(path)
    logger.info(f"binary file loaded from : {path}")
    return data