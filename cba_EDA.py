import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import openai
import os

# Set your OpenAI API key (make sure it's in your environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Agentic EDA Tool with AI Insights")
st.write("Upload a CSV file to begin automated exploratory data analysis and interact with AI for deeper insights.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- BASIC EDA ---
    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Basic Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Summary Statistics (Numeric Columns Only)")
    numeric_df = df.select_dtypes(include=['number'])
    st.write(numeric_df.describe())

    st.subheader("Correlation Matrix (Numeric Columns Only)")
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader("Categorical Feature Analysis")
    categorical_df = df.select_dtypes(include=['object', 'category'])
    for col in categorical_df.columns:
        st.write(f"Value Counts for {col}:")
        st.write(df[col].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- AI INSIGHTS ---
    st.subheader("AI-Powered Data Insights")

    # Prepare a compact summary of data issues for the LLM
    invalid_summary = {
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": numeric_df.describe().to_dict() if not numeric_df.empty else "No numeric data",
        "categorical_columns": {col: df[col].unique().tolist()[:10] for col in categorical_df.columns}
    }

    user_question = st.text_input("Ask AI about data issues (e.g., 'What looks invalid in this dataset?')")

    if st.button("Analyze with AI") or user_question:
        context = f"""
        Dataset Issues Summary:
        - Missing Values: {invalid_summary['missing_values']}
        - Numeric Summary: {invalid_summary['numeric_summary']}
        - Example Categorical Values: {invalid_summary['categorical_columns']}
        """

        messages = [
            {"role": "system", "content": "You are a senior data analyst. Identify invalid data, outliers, and inconsistencies. Suggest fixes (e.g., imputation, removal, normalization)."},
            {"role": "user", "content": context},
        ]

        if user_question:
            messages.append({"role": "user", "content": user_question})

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500
            )
            reply = response["choices"][0]["message"]["content"]
            st.write("### AI Analysis")
            st.write(reply)

        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")

    st.success("EDA completed. Use AI analysis for deeper insights.")

    # Option to re-upload corrected dataset
    st.subheader("Upload Corrected Data for Comparison")
    corrected_file = st.file_uploader("Upload a corrected CSV file", type="csv", key="corrected")
    if corrected_file:
        corrected_df = pd.read_csv(corrected_file)
        st.write("### Corrected Data Preview")
        st.dataframe(corrected_df.head())
        st.write("Differences in missing values between original and corrected:")
        diff = df.isnull().sum() - corrected_df.isnull().sum()
        st.write(diff)
