# End to End Data Science project



## workflows ML Pipeline 

1. Data ingetion 
2. Data validation
3. Data Transformation 
4. Model trainer 
5. Model evaluation  



## Workflows 

1. update config.yaml 
2. update schema.yaml 
3. update params.yaml 
4. update entity 
5. update the configration manager in src config 
6. update the component 
7. update the pipeline 
8. update the main.py 




Data ingetion :
    1. update config.yaml o
      - data ingestion pipeline require input so we crete those input here :
      - also update constant file -> __init__.py 

    2. update schema.yaml 
    3. update params.yaml 
    4. update entity 
      - config_entity.py 
    5. update the config  manager in src configaration.py

    6. update the component 
    - create the file : components-> data_ingestion.py 
    7. update the pipeline 
     - pipeline-> data_ingestion_pipeline.py 
     - update the pipeline code 
     

    8. update the main.py 
      - finally run this the all pipeline from main.py  







2. Data validation : 
   whenever we get new data for prediction  : those features should no tbe change and their dtypes also should not have to change here we will vlidate decah and every features 


  1. update config.yaml 
      upadate config.yaml
  2. update schema.yaml :
      upadte schemas.yaml (update every features ans their data types )
  3. update params.yaml 
  4. update entity 
  5. update the configration manager in src config 
  6. update the component 
  7. update the pipeline 
  8. update the main.py 

