import os
import streamlit as st
from eda.loader import load_data
from eda.analysis import perform_eda
from eda.plots import generate_plots
from eda.scoring import compute_data_quality
from ui.layout import create_layout
from ui.controls import create_controls
from ui.chat import chat_interface
from ai.azure_openai import initialize_openai_client

def main():
    st.set_page_config(page_title="Agentic EDA Tool", layout="wide")
    
    # Initialize Azure OpenAI client
    openai_client = initialize_openai_client()
    
    # Create the layout for the app
    create_layout()
    
    # Create controls for user interaction
    create_controls()
    
    # File uploader for CSV files
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            df = load_data(uploaded_file)
            if df is not None:
                perform_eda(df)
                generate_plots(df)
                dq_score = compute_data_quality(df)
                st.metric("Data Quality Score", f"{dq_score:.2f}/100")
    
    # Chat interface for discussing EDA results
    chat_interface(openai_client)

if __name__ == "__main__":
    main()