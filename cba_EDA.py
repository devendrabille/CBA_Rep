import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
from openai import OpenAI

# =======================
# Set OpenAI API key
# =======================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =======================
# Streamlit App Layout
# =======================
st.title("🤖 Agentic EDA Tool with Chatbot")
st.write("Upload a CSV file to begin automated exploratory data analysis and interact with the data using AI.")

# =======================
# File Uploader
# =======================
uploaded_file = st.file_uploader("📂 Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -----------------------
    # Data Preview
    # -----------------------
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # -----------------------
    # Basic Info
    # -----------------------
    st.subheader("📋 Basic Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # -----------------------
    # Missing Values
    # -----------------------
    st.subheader("⚠️ Missing Values")
    st.write(df.isnull().sum())

    # -----------------------
    # Summary Statistics
    # -----------------------
    st.subheader("📊 Summary Statistics (Numeric Columns Only)")
    numeric_df = df.select_dtypes(include=['number'])
    st.write(numeric_df.describe())

    # -----------------------
    # Correlation Matrix
    # -----------------------
    st.subheader("📈 Correlation Matrix (Numeric Columns Only)")
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # -----------------------
    # Categorical Feature Analysis
    # -----------------------
    st.subheader("🔠 Categorical Feature Analysis")
    categorical_df = df.select_dtypes(include=['object', 'category'])
    for col in categorical_df.columns:
        st.write(f"Value Counts for **{col}**:")
        st.write(df[col].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # -----------------------
    # Special Columns (if exist)
    # -----------------------
    if 'price' in df.columns:
        st.subheader("💰 Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['price'], kde=True, ax=ax)
        st.pyplot(fig)

    if 'size' in df.columns and 'price' in df.columns:
        st.subheader("📏 Price vs Size")
        fig, ax = plt.subplots()
        sns.scatterplot(x='size', y='price', data=df, ax=ax)
        st.pyplot(fig)

    if 'age' in df.columns and 'price' in df.columns:
        st.subheader("📅 Price vs Age")
        fig, ax = plt.subplots()
        sns.scatterplot(x='age', y='price', data=df, ax=ax)
        st.pyplot(fig)

    if 'bedrooms' in df.columns and 'price' in df.columns:
        st.subheader("🛏 Price vs Bedrooms")
        fig, ax = plt.subplots()
        sns.boxplot(x='bedrooms', y='price', data=df, ax=ax)
        st.pyplot(fig)

    st.success("✅ EDA completed. Scroll through the sections above to view insights.")

    # =======================
    # Chatbot Interface
    # =======================
    st.subheader("💬 Ask Questions About Your Data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask a question about the dataset")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Prepare context from dataset
        context = f"Summary statistics:\n{numeric_df.describe().to_string()}\n\nMissing values:\n{df.isnull().sum().to_string()}"

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst assistant. Answer questions based on the dataset summary."},
                    {"role": "user", "content": context},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=500
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)
