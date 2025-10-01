import os 
from pathlib import Path 
import logging

logging.basicConfig(level=logging.INFO)

list_of_file = {

    "src/__init__.py" ,
    "src/data_ingestion.py" , 
    "src/exceptions.py" , 
    "src/utils/ __init__.py" , 
    "src/utils/logger.py" , 
    "src/modeling.py" , 
    "data/raw_peper.csv" ,
    "app.py" , 
    "Readme.md" ,
    "requirements.txt" , 
    "setup.py"

}

for files in list_of_file : 
    file_path = Path(files)
    file_dir , file_name = os.path.split(files)

    if file_dir != "" : 
        os.makedirs(file_dir , exist_ok = True)
        logging.info(f"Creating directory : {file_dir} for the file : {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0) :
        with open (file_path , 'w') as f :
            pass 
        logging.info(f"Creating Empty file , {file_path}")

    else :
        logging.info(f"File already exist {file_path}")

