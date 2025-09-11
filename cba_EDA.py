import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
from openai import OpenAI
from fpdf import FPDF

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    missing_vals = df.isnull().sum()
    st.write(missing_vals)

    st.subheader("Summary Statistics (Numeric Columns Only)")
    numeric_df = df.select_dtypes(include=['number'])
    st.write(numeric_df.describe())

    st.subheader("Correlation Matrix (Numeric Columns Only)")
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # --- OUTLIER DETECTION ---
    st.subheader("Outlier Detection (IQR Method)")
    outlier_summary = {}
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = numeric_df[(numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)]
        outlier_summary[col] = len(outliers)
        fig, ax = plt.subplots()
        sns.boxplot(x=numeric_df[col], ax=ax)
        st.pyplot(fig)
    st.write("Outlier counts per column:", outlier_summary)

    # --- DATA QUALITY SCORE ---
    st.subheader("Data Quality Score")
    total_cells = df.shape[0] * df.shape[1]
    missing_ratio = missing_vals.sum() / total_cells
    outlier_ratio = sum(outlier_summary.values()) / numeric_df.shape[0] if not numeric_df.empty else 0
    quality_score = 100 - (missing_ratio * 50 + outlier_ratio * 50)
    st.write(f"Estimated Data Quality Score: {quality_score:.2f}/100")

    # --- CATEGORICAL ANALYSIS ---
    st.subheader("Categorical Feature Analysis")
    categorical_df = df.select_dtypes(include=['object', 'category'])
    for col in categorical_df.columns:
        st.write(f"Value Counts for {col}:")
        st.write(df[col].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- AI CHART EXPLANATION ---
    st.subheader("AI Chart Explanation")
    chart_question = st.text_input("Ask AI to explain a chart or pattern")
    if st.button("Explain Chart") and chart_question:
        chart_context = f"Numeric Summary: {numeric_df.describe().to_dict()}\nOutlier Summary: {outlier_summary}"
        messages = [
            {"role": "system", "content": "You are a data analyst. Explain the chart and patterns in simple terms."},
            {"role": "user", "content": chart_context},
            {"role": "user", "content": chart_question}
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500
            )
            reply = response.choices[0].message.content
            st.write("### AI Chart Explanation")
            st.write(reply)
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")

    # --- INSIGHT CARDS ---
    st.subheader("AI Insight Cards")
    insight_context = f"Missing Values: {missing_vals.to_dict()}\nOutliers: {outlier_summary}\nNumeric Summary: {numeric_df.describe().to_dict()}"
    messages = [
        {"role": "system", "content": "You are a data analyst. Generate 3 key insights from the dataset."},
        {"role": "user", "content": insight_context}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300
        )
        insights = response.choices[0].message.content.split("\n")
        for insight in insights:
            st.info(insight)
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")

    # --- AUTO SUGGESTIONS ---
    st.subheader("Auto Suggestions for Fixes")
    suggestion_context = f"Missing Values: {missing_vals.to_dict()}\nOutliers: {outlier_summary}"
    messages = [
        {"role": "system", "content": "You are a data scientist. Suggest preprocessing steps to clean the data."},
        {"role": "user", "content": suggestion_context}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300
        )
        suggestions = response.choices[0].message.content.split("\n")
        for suggestion in suggestions:
            st.warning(suggestion)
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")

    # --- REPORT GENERATION ---
    st.subheader("Downloadable Report")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="EDA Report", ln=True, align='C')
        pdf.multi_cell(0, 10, txt=f"Missing Values:\n{missing_vals.to_string()}\n\nOutliers:\n{outlier_summary}\n\nData Quality Score: {quality_score:.2f}/100")
        report_path = "eda_report.pdf"
        pdf.output(report_path)
        with open(report_path, "rb") as f:
            st.download_button("Download Report", f, file_name="eda_report.pdf")

    st.success("EDA completed with advanced insights and suggestions.")
