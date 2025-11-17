import pandas as pd
import streamlit as st

def load_csv(file):
    try:
        df = pd.read_csv(file, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def validate_data(df):
    if df is None:
        return False
    if df.empty:
        st.warning("The uploaded file is empty.")
        return False
    return True

def handle_file_upload():
    uploaded_file = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=False)
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if validate_data(df):
            st.success("File loaded successfully!")
            return df
    return None