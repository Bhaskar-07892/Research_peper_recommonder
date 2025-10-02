import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from src.exceptions import CustomException
from src.utils.logger import logging 

ARXIV_API_URL = "http://export.arxiv.org/api/query"
QUERY = 'cat:cs.AI OR cat:cs.LG' 
MAX_RESULTS = 500
OUTPUT_FILE = os.path.join('data', 'raw_papers.csv')

class DataIngestion:
    def __init__ (self) :
        
        self.output_dir = os.path.dirname(OUTPUT_FILE)
        if not os.path.exists(self.output_dir) : 
            os.makedirs(self.output_dir)
            logging.info(f"Creating directory {self.output_dir}.") 

    def fetch_data (self) -> str :
        params = {
            'search_query' : QUERY ,
            'start' : 0 , 
            'max_results' : MAX_RESULTS 
        }
        logging.info(f"Fetching data from ArXiv API for query: {QUERY}")

        try : 
            response = requests.get(url = ARXIV_API_URL , params=params) 
            response.raise_for_status()
            logging.info(f"Successfully fetched : {MAX_RESULTS} papers.") 
            return response.text
        
        except requests.exceptions.RequestException as e :
            logging.error(f"API request failed : {e} ")
            raise CustomException(e)


    def parse_arxiv_xml (self , xml_data:str) -> pd.DataFrame :
        logging.info("Starting XML parsing.")
        data = []

        try : 
            root = ET.fromstring(xml_data)

            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                published = entry.find('{http://www.w3.org/2005/Atom}published').text.strip()
                
                authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]

                data.append({
                    'id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1],
                    'title': title,
                    'summary': summary,
                    'published': published,
                    'authors': ', '.join(authors)
                })
            
            df = pd.DataFrame(data) 
            logging.info(f"Successfully parsed {len(df)} papers into DataFrame.")
            return df
    
        except Exception as e :
            logging.error(f"Error in parsing logic {e}") 
            raise CustomException (e)
        

    def save_data (self, df: pd.DataFrame): 
        logging.info(f"Saving data to {OUTPUT_FILE}")
        try:
            df.to_csv(OUTPUT_FILE, index=False)
            logging.info("Data saved successfully.")
        except IOError as e:
            logging.error(f"File Save Failed: {e}")
            raise CustomException(f"Error saving data to CSV: {e}")
        
    def initiate_data_ingestion(self):
        
        try:
            # 1. pulling data
            xml_data = self.fetch_data()
            
            # 2. parsing data
            df = self.parse_arxiv_xml(xml_data)
            
            # 3. save data
            self.save_data(df)
            
            logging.info("Data Ingestion Pipeline completed successfully.")
            return OUTPUT_FILE

        except CustomException as e:
            logging.error(f"Data Ingestion failed with custom error: {e}")

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()