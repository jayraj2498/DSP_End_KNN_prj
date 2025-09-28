from src.datascience import logger 
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_evaluation import ModelEvaluation
from src.datascience.constants import * 



STAGE_NAME = "Model Evaluation Stage" 

class ModelEvaluationTrainingPipeline :
    def __init__(self):
        pass 
    
    
    def initiate_model_model_evaluation(self) :
        
        config = ConfigurationManager() 
        model_eval_config = config.get_model_evaluation_config() 
                
        model_eval = ModelEvaluation(config=model_eval_config) 
        model_eval.log_into_mlflow() 
        
    
                
                
           
            
       