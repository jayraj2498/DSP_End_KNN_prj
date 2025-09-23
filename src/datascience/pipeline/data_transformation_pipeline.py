from src.datascience.components.data_transformation import DataTransformation 
from src.datascience.config.configuration import ConfigurationManager
from src.datascience import logger 
from pathlib import Path 


STAGE_NAME = "Data Transformation Stage" 

class DataTransformationTrainingPipeline: 
    def __init__(self): 
        pass 
    
    def initiate_data_transformation(self): 
        try:
            # Read validation status
            with open(Path("artifacts/data_validation/status.txt"), 'r') as f:
                status = f.read().lower().strip()

            # Check if status contains "true"
            if "true" in status:
                logger.info(" Data schema is valid, proceeding with transformation.")
                config = ConfigurationManager() 
                data_transformation_config = config.get_data_transformation_config() 
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_splitting() 
            else:
                raise Exception("your data schema is not valid ") 
                
        except Exception as e:
            logger.exception(e)
            raise e



