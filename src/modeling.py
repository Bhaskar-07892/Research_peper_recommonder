import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

from src.exceptions import CustomException
from src.utils.logger import logging

# ----------------------------------------------------------------------
# 1. Data Loading and Processing
# ----------------------------------------------------------------------

def load_and_process_data(filepath: str) -> pd.DataFrame:
    
    logging.info(f"Loading and processing data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        
        # combine 'title' and 'summary' 
        df['combined_text'] = df['title'].fillna('') + ' ' + df['summary'].fillna('')
        
        df.drop('summary', axis=1, inplace=True)
        
        df.dropna(subset=['combined_text', 'published'], inplace=True)
        
        # change 'published' coloumn to datetime obj 
        df['published_dt'] = pd.to_datetime(df['published'], errors='coerce')
        df.dropna(subset=['published_dt'], inplace=True)
        
        logging.info(f"Data processing complete. {len(df)} valid papers remaining.")
        return df
    
    except FileNotFoundError:
        logging.error(f"Data file not found at {filepath}")
        raise CustomException(f"Modeling data not found. Run data ingestion first.")
    except Exception as e:
        logging.error(f"Error during data loading/processing: {e}")
        raise CustomException(f"Error in data processing: {e}")

# ----------------------------------------------------------------------
# 2. Core ML Modeling - TF-IDF and Cosine Similarity
# ----------------------------------------------------------------------

def create_similarity_matrix(df: pd.DataFrame):
    
    logging.info("Creating TF-IDF vectors and Cosine Similarity matrix...")
    
    tfidf = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 2),   
        min_df=3              
    )
    
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    logging.info("Similarity matrix successfully created.")
    return cosine_sim

# ----------------------------------------------------------------------
# 3. Hybrid Logic and Recommendation Function
# ----------------------------------------------------------------------

def calculate_date_boost(published_dt: datetime) -> float:
    
    today = datetime.now().date()
    pub_date = published_dt.date()
    days_since_pub = (today - pub_date).days
    
    if days_since_pub > 1095:
        return 0.0
    
    # (1 - (days_since_pub / 1095)) * 0.05
    boost = (1 - (days_since_pub / 1095)) * 0.05
    return max(0.0, boost)

def recommend_papers(df: pd.DataFrame, cosine_sim: np.ndarray, index: int, top_n: int = 10) -> pd.DataFrame:
    
    try:
        sim_scores = list(enumerate(cosine_sim[index]))
        
        hybrid_scores = []
        for i, score in sim_scores:
            if i != index:
                date_boost = calculate_date_boost(df.iloc[i]['published_dt'])
                final_score = score + date_boost
                hybrid_scores.append((i, final_score))
        
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
        
        top_indices = [i[0] for i in hybrid_scores[:top_n]]
        
        logging.info(f"Generated {len(top_indices)} hybrid recommendations for index {index}.")
        
        recommendations_df = df.iloc[top_indices].copy()
        
        final_scores_map = {idx: score for idx, score in hybrid_scores}
        recommendations_df['Hybrid Score'] = recommendations_df.index.map(lambda i: final_scores_map.get(i, 0))
        
        return recommendations_df[['title', 'authors', 'published', 'Hybrid Score']]
    
    except IndexError:
        logging.error(f"Index {index} out of bounds for the DataFrame.")
        raise CustomException(f"Paper index {index} not found in processed data.")
    except Exception as e:
        logging.error(f"Error during recommendation logic: {e}")
        raise CustomException(f"An unexpected error occurred during modeling: {e}")

# ----------------------------------------------------------------------
# 4. Main Execution Block - For testing or standalone execution
# ----------------------------------------------------------------------

if __name__ == '__main__':
    try:
        processed_df = load_and_process_data('data/raw_papers.csv')
        
        sim_matrix = create_similarity_matrix(processed_df)
        
        target_paper_title = processed_df.iloc[0]['title']
        logging.info(f"--- Recommendations for: {target_paper_title} ---")
        
        recommendations = recommend_papers(processed_df, sim_matrix, 0, top_n=5)
        print(recommendations)
        
    except CustomException as e:
        logging.error(f"Modeling process failed: {e}")
    except Exception as e:
        logging.critical(f"A CRITICAL error occurred in the modeling script: {e}")
