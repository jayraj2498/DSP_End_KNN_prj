from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience import logger  



STAGE_NAME = "Model Trainer Stage" 

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass 
    
    
    def initiate_model_training(self) :
        config =ConfigurationManager() 
        mode_trainer_config =config.get_model_trainer_config() 
        model_trainer_config = ModelTrainer(config=mode_trainer_config) 
        model_trainer_config.train()

