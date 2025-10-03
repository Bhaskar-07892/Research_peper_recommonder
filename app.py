import streamlit as st
import pandas as pd
import numpy as np
import os

try:
    from src.modeling import load_and_process_data, create_similarity_matrix, recommend_papers
    from src.exceptions import CustomException
    from src.utils.logger import logging
except ImportError as e:
    st.error(f"Module import error: {e}. Please ensure you are running this from the project root and all dependencies are installed.")
    st.stop()


# ----------------------------------------------------------------------
# 1. Data Loading and Modeling (Caching for efficiency)
# Use @st.cache_data to tell Streamlit that this function
# should run only once, even if the user interacts with the UI.
# ----------------------------------------------------------------------

@st.cache_data
def load_and_process_assets(data_filepath='data/raw_papers.csv'):
    """
    Loads and processes data, and creates the similarity matrix. 
    This function will run only once.
    """
    logging.info("Starting asset loading: Data & Model.")
    
    if not os.path.exists(data_filepath):
        raise CustomException(f"Data file not found at {data_filepath}. Please run data ingestion.")

    # 1. Load and process the data
    processed_df = load_and_process_data(data_filepath)
    
    # 2. Create the similarity matrix
    sim_matrix = create_similarity_matrix(processed_df)
    
    logging.info("Assets loaded and Similarity Matrix created successfully.")
    return processed_df, sim_matrix

# ----------------------------------------------------------------------
# 2. Streamlit UI (Web Application Interface)
# ----------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Academic Paper Recommender",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📚 ArXiv Hybrid Paper Recommendation System")
    st.markdown("Select any paper title and we will provide the **5** most relevant recommendations based on text similarity and publication date (Date Boost).")

    try:
        # Load data and matrix from the cached function
        df, cosine_sim = load_and_process_assets()
        
    except CustomException as e:
        st.error(f"CRITICAL ERROR: {e}")
        st.info("Please ensure the data ingestion step was run successfully and 'data/raw_papers.csv' exists.")
        return # Stop the app if data cannot be loaded

    # --- UI Components ---
    
    # 1. Prepare paper titles dropdown for selection
    paper_titles = df['title'].tolist()
    
    # Let the user select a paper
    selected_title = st.selectbox(
        "Select a paper to get recommendations:",
        paper_titles,
        index=0 # Select the first item by default
    )

    # 2. Button to generate recommendations
    if st.button("Generate Recommendations", type="primary"):
        if selected_title:
            with st.spinner('Generating recommendations...'):
                
                # Find the index of the selected title
                selected_index = df[df['title'] == selected_title].index[0]
                
                # Call the model function, change top_n from 10 to 5
                recommendations_df = recommend_papers(df, cosine_sim, selected_index, top_n=5)
                
                st.subheader(f"🔍 Top 5 Recommendations for: _{selected_title}_")
                st.markdown("---") # For UI separation
                
                # Display results - use st.expander to show summary
                for index, row in recommendations_df.iterrows():
                    title = row['title']
                    authors = row['authors']
                    published = row['published']
                    hybrid_score = f"{row['Hybrid Score']:.4f}"
                    
                    # Use the title to find content from the original df
                    paper_data = df[df['title'] == title].iloc[0]
                    # 'combined_text' includes both title and summary
                    summary_text = paper_data['combined_text'] 
                    
                    # NEW: Get Paper ID and create ArXiv link
                    paper_id = paper_data['id']
                    arxiv_link = f"https://arxiv.org/abs/{paper_id}"


                    expander_title = f"📄 **{title}** | Authors: {authors} | Published: {published} | Score: {hybrid_score}"
                    with st.expander(expander_title):
                        st.markdown("### Paper Summary")
                        st.info(summary_text)
                        
                        # NEW: Display ArXiv ID and direct link
                        st.markdown(f"**ArXiv ID:** `{paper_id}` | [**View PDF/Abstract on ArXiv ↗️**]({arxiv_link})")

        else:
            st.warning("Please select a paper title to generate recommendations.")

if __name__ == '__main__':
    main()
