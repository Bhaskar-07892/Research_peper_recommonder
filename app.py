import streamlit as st
import pandas as pd
import os
import numpy as np

# --- Import custom modules ---
try:
    from src.modeling import load_and_process_data, create_similarity_matrix, recommend_papers
    from src.exceptions import CustomException
    from src.utils.logger import logging
except ImportError as e:
    # A simple way to handle dependency error during deployment
    st.error(f"Module import error: {e}. Please ensure you have the full project structure (src/) and all dependencies.")
    st.stop()


# ----------------------------------------------------------------------
# 1. Data Loading and Modeling (Caching for efficiency)
# Use @st.cache_data to ensure this expensive function runs only once.
# ----------------------------------------------------------------------

@st.cache_data
def load_and_process_assets(data_filepath='data/raw_papers.csv'):
    """
    Loads and processes data, and creates the similarity matrix.
    This function will run only once.
    """
    logging.info("Starting asset loading: Data & Model.")
    
    if not os.path.exists(data_filepath):
        # Raise an error if data is missing (crucial for deployment)
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
        page_title="ArXiv Paper Recommender",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # --- Header and Introduction ---
    st.markdown("<h1 style='text-align: center; color: #1e88e5;'>📚 ArXiv Hybrid Paper Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Select any paper title and we will provide the **5** most relevant recommendations based on **Text Similarity** and a **Date Boost** (prioritizing newer research).")
    
    try:
        # Load data and matrix from the cached function
        df, cosine_sim = load_and_process_assets()
        
    except CustomException as e:
        st.error(f"CRITICAL ERROR: {e}")
        st.info("Please ensure the data ingestion step was run successfully and 'data/raw_papers.csv' exists.")
        return # Stop the app if data cannot be loaded

    # --- Input Section ---
    
    # Use columns to make the input layout cleaner
    input_col, button_col = st.columns([4, 1])
    
    paper_titles = df['title'].tolist()
    
    with input_col:
        selected_title = st.selectbox(
            "Select a paper to get recommendations:",
            paper_titles,
            index=0
        )

    # State variable to track if recommendations have been generated
    if 'recommendations_generated' not in st.session_state:
        st.session_state.recommendations_generated = False

    with button_col:
        # Add some vertical space to align the button
        st.write("")
        st.write("") 
        if st.button("Generate Recommendations", type="primary"):
            st.session_state.recommendations_generated = True

    # --- Recommendation Logic and Display ---
    if st.session_state.recommendations_generated:
        if selected_title:
            with st.spinner('Generating recommendations... This may take a moment to load the model.'):
                
                # Find the index of the selected title
                selected_index = df[df['title'] == selected_title].index[0]
                
                # Call the model function
                recommendations_df = recommend_papers(df, cosine_sim, selected_index, top_n=5)
                
                # --- Selected Paper Detail (Better UI for Input) ---
                
                # Get details for the selected paper
                selected_paper_data = df[df['title'] == selected_title].iloc[0]
                selected_authors = selected_paper_data['authors']
                selected_published = selected_paper_data['published']
                selected_summary = selected_paper_data['combined_text']
                
                # Display the selected paper in a clean expander/box
                with st.expander(f"📖 **Selected Paper Details: {selected_title}**", expanded=False):
                    st.markdown(f"**Authors:** _{selected_authors}_")
                    st.markdown(f"**Published Date:** _{selected_published}_")
                    st.markdown("---")
                    st.markdown(f"**Summary:** {selected_summary}")
                
                st.markdown("---")
                st.subheader(f"✅ Top 5 Recommendations")
                st.write(f"The best matches for **{selected_title}** are listed below:")

                # --- Display Results Loop (Cleaner UI) ---
                
                # Loop through the recommendations and display them beautifully
                for index, row in recommendations_df.iterrows():
                    title = row['title']
                    authors = row['authors']
                    published = row['published']
                    hybrid_score = f"{row['Hybrid Score']:.4f}"
                    
                    paper_data = df[df['title'] == title].iloc[0]
                    summary_text = paper_data['combined_text'] 
                    paper_id = paper_data['id']
                    arxiv_link = f"https://arxiv.org/abs/{paper_id}"

                    # Use a container for visual separation of each recommendation block
                    with st.container(border=True):
                        
                        # 1. Row for Title and Score
                        title_col, score_col = st.columns([4, 1])
                        
                        with title_col:
                            st.markdown(f"#### 📄 {title}")
                        
                        with score_col:
                            # Display score prominently as a metric
                            st.metric(label="Hybrid Score", value=hybrid_score)
                        
                        # 2. Row for Metadata (Authors, Date, Link)
                        meta_col1, meta_col2 = st.columns([1, 2])
                        
                        with meta_col1:
                            st.markdown(f"**Published:** {published}")
                            st.markdown(f"**ArXiv ID:** `{paper_id}`")
                        
                        with meta_col2:
                            st.markdown(f"**Authors:** _{authors}_")
                            # Direct link to the source
                            st.markdown(f"[**View Full Paper/PDF on ArXiv ↗️**]({arxiv_link})")

                        # 3. Summary Expander
                        with st.expander("Expand to Read Abstract"):
                            st.markdown(f"**Abstract:** {summary_text}")
                            
            
        else:
            st.warning("Please select a paper title to generate recommendations.")

if __name__ == '__main__':
    main()
