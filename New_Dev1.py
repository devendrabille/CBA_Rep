import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
from fpdf import FPDF

# === NEW: Ollama client ===
try:
    import ollama
except ImportError:
    ollama = None

st.set_page_config(page_title="Agentic EDA Tool with Local AI (Ollama)", layout="wide")
st.title("Agentic EDA Tool with AI Insights (Local LLM via Ollama)")
st.write("Upload a CSV file to begin automated exploratory data analysis and interact with a **local** AI (no tokens needed).")

# --------------------------
# Sidebar: Model & Params
# --------------------------
st.sidebar.header("AI Model Settings (Ollama)")
if ollama is None:
    st.sidebar.error("`ollama` package not found. Run: `pip install ollama`")
    st.stop()

# Try to list installed models from Ollama
try:
    available_models = [m["model"] for m in ollama.list().get("models", [])]
except Exception as e:
    available_models = []
    st.sidebar.error(
        "Could not reach Ollama server. Ensure Ollama is installed and running.\n"
        "On Linux: run `ollama serve`. On macOS/Windows it usually runs automatically."
    )

default_model = "llama3:8b" if "llama3:8b" in available_models else (available_models[0] if available_models else "llama3:8b")
model_name = st.sidebar.selectbox("Select model", options=[default_model] + [m for m in available_models if m != default_model])
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.3, 0.05)
max_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 400, 16)
system_role = st.sidebar.text_area(
    "System prompt",
    value="You are a helpful data analyst. Explain EDA findings clearly in plain language, with practical suggestions.",
    height=100
)

# Helper to call Ollama chat
def call_ollama_chat(model, system_prompt, user_messages, temperature=0.3, max_tokens=400):
    """
    user_messages: list[str] - will be appended as user turns.
    Returns: str response content.
    """
    if ollama is None:
        raise RuntimeError("ollama package not available")

    messages = [{"role": "system", "content": system_prompt}]
    for um in user_messages:
        messages.append({"role": "user", "content": um})

    try:
        res = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return res.get("message", {}).get("content", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama chat error: {e}")

# --------------------------
# File Uploader
# --------------------------
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # --- BASIC EDA ---
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Basic Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    st.text(info_text)

    st.subheader("Missing Values")
    missing_vals = df.isnull().sum()
    st.write(missing_vals)

    st.subheader("Summary Statistics (Numeric Columns Only)")
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        st.write(numeric_df.describe())
    else:
        st.info("No numeric columns detected.")

    # --- CORRELATION ---
    st.subheader("Correlation Matrix (Numeric Columns Only)")
    if not numeric_df.empty and numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(min(0.6 * corr.shape[0] + 4, 12), min(0.6 * corr.shape[1] + 4, 12)))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need at least 2 numeric columns for a correlation matrix.")

    # --- OUTLIER DETECTION (IQR) ---
    st.subheader("Outlier Detection (IQR Method)")
    outlier_summary = {}
    total_numeric_cells = int(numeric_df.shape[0] * numeric_df.shape[1]) if not numeric_df.empty else 0

    if not numeric_df.empty:
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)]
            outlier_summary[col] = int(outliers.shape[0])

            fig, ax = plt.subplots(figsize=(6, 1.8))
            sns.boxplot(x=numeric_df[col], ax=ax, color="#5DADE2")
            ax.set_title(f"Boxplot: {col}")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Skipping outlier detection since no numeric columns found.")

    st.write("Outlier counts per column:", outlier_summary)

    # --- DATA QUALITY SCORE ---
    st.subheader("Data Quality Score")
    total_cells = int(df.shape[0] * df.shape[1]) if df.size else 0
    missing_ratio = (missing_vals.sum() / total_cells) if total_cells else 0.0

    # Normalize outlier ratio over numeric cells (prevents inflated score on wide data)
    outlier_count_total = sum(outlier_summary.values())
    outlier_ratio = (outlier_count_total / total_numeric_cells) if total_numeric_cells else 0.0

    # Weighted simple score
    quality_score = max(0.0, 100 - (missing_ratio * 50 + outlier_ratio * 50))
    st.write(f"Estimated Data Quality Score: `{quality_score:.2f}/100`")
    with st.expander("Quality score details"):
        st.write({
            "total_cells": total_cells,
            "missing_ratio": round(missing_ratio, 4),
            "total_numeric_cells": total_numeric_cells,
            "outlier_ratio": round(outlier_ratio, 4)
        })

    # --- CATEGORICAL ANALYSIS ---
    st.subheader("Categorical Feature Analysis")
    categorical_df = df.select_dtypes(include=["object", "category"])
    if not categorical_df.empty:
        max_categories_to_plot = st.slider("Max categories to plot per feature (top N)", 5, 50, 20, 1)
        for col in categorical_df.columns:
            st.write(f"**Value Counts for {col}:**")
            st.write(df[col].value_counts().head(50))  # show top 50 in table

            top_vals = df[col].value_counts().head(max_categories_to_plot)
            top_categories = list(top_vals.index)
            df_plot = df[df[col].isin(top_categories)]

            fig, ax = plt.subplots(figsize=(min(0.4 * len(top_categories) + 4, 14), 4))
            sns.countplot(x=col, data=df_plot, order=top_categories, ax=ax, palette="viridis")
            plt.xticks(rotation=45, ha="right")
            ax.set_title(f"Top {len(top_categories)} Categories: {col}")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("No categorical columns detected.")

    # =====================================================
    # AI SECTIONS (Ollama)
    # =====================================================

    # --- AI CHART EXPLANATION ---
    st.subheader("AI Chart Explanation (Local LLM)")
    chart_question = st.text_input("Ask AI to explain a chart or pattern (e.g., 'Explain the correlations' or 'What do the outliers mean?')")
    if st.button("Explain Chart") and chart_question:
        # Keep context concise to avoid long prompts
        num_summary = numeric_df.describe().to_dict() if not numeric_df.empty else {}
        top_missing = missing_vals.sort_values(ascending=False).head(10).to_dict()
        context = (
            f"High-level context for the dataset:\n"
            f"- Missing values (top 10): {top_missing}\n"
            f"- Outlier counts per numeric column: {outlier_summary}\n"
            f"- Numeric summary (describe): {num_summary}\n"
            f"User question: {chart_question}"
        )

        with st.spinner("Thinking with local model..."):
            try:
                reply = call_ollama_chat(
                    model=model_name,
                    system_prompt=system_role,
                    user_messages=[context],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                st.write("### AI Chart Explanation")
                st.write(reply)
            except Exception as e:
                st.error(str(e))

    # --- AI INSIGHT CARDS ---
    st.subheader("AI Insight Cards")
    insight_context = (
        f"Dataset size: {df.shape}\n"
        f"Missing values (top 10): {missing_vals.sort_values(ascending=False).head(10).to_dict()}\n"
        f"Outliers per numeric column: {outlier_summary}\n"
        f"Numeric summary (describe): {(numeric_df.describe().to_dict() if not numeric_df.empty else {})}\n"
        "Generate 3-5 concise, high-signal insights about the data. Use bullets."
    )
    if st.button("Generate Insights"):
        with st.spinner("Generating insights with local model..."):
            try:
                resp = call_ollama_chat(
                    model=model_name,
                    system_prompt="You are a data analyst. Provide crisp, insightful, and actionable observations.",
                    user_messages=[insight_context],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Display each bullet as a card
                for line in [l for l in resp.split("\n") if l.strip()]:
                    st.info(line.strip("•- "))
            except Exception as e:
                st.error(str(e))

    # --- AUTO SUGGESTIONS ---
    st.subheader("Auto Suggestions for Fixes")
    suggestion_context = (
        f"Missing Values: {missing_vals.to_dict()}\n"
        f"Outliers: {outlier_summary}\n"
        "Suggest concrete preprocessing steps (imputation, encoding, scaling, outlier handling, feature engineering). "
        "Be specific to the issues detected."
    )
    if st.button("Suggest Preprocessing"):
        with st.spinner("Drafting suggestions..."):
            try:
                resp = call_ollama_chat(
                    model=model_name,
                    system_prompt="You are a data scientist. Suggest practical preprocessing steps.",
                    user_messages=[suggestion_context],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for line in [l for l in resp.split("\n") if l.strip()]:
                    st.warning(line.strip("•- "))
            except Exception as e:
                st.error(str(e))

    # --- REPORT GENERATION ---
    st.subheader("Downloadable Report")
    include_ai_summary = st.checkbox("Include brief AI-driven summary in report", value=True)
    if st.button("Generate PDF Report"):
        ai_summary = ""
        if include_ai_summary:
            quick_context = (
                f"Shape: {df.shape}, Missing (top 5): {missing_vals.sort_values(ascending=False).head(5).to_dict()}, "
                f"Outliers: {outlier_summary}. Provide a succinct executive summary (4-6 lines)."
            )
            try:
                ai_summary = call_ollama_chat(
                    model=model_name,
                    system_prompt="You are a senior data analyst. Write a crisp executive summary.",
                    user_messages=[quick_context],
                    temperature=0.2,
                    max_tokens=220,
                )
            except Exception as e:
                ai_summary = f"(AI summary unavailable: {e})"

        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(0, 10, txt="EDA Report", ln=True, align="C")
            pdf.set_font("Arial", size=11)

            pdf.ln(3)
            pdf.multi_cell(0, 6, txt="--- BASIC INFO ---")
            pdf.multi_cell(0, 6, txt=info_text)

            pdf.ln(2)
            pdf.multi_cell(0, 6, txt="--- MISSING VALUES ---")
            pdf.multi_cell(0, 6, txt=missing_vals.to_string())

            pdf.ln(2)
            pdf.multi_cell(0, 6, txt="--- OUTLIERS ---")
            pdf.multi_cell(0, 6, txt=str(outlier_summary))

            pdf.ln(2)
            pdf.multi_cell(0, 6, txt="--- DATA QUALITY ---")
            pdf.multi_cell(0, 6, txt=f"Quality Score: {quality_score:.2f}/100")

            if include_ai_summary:
                pdf.ln(2)
                pdf.multi_cell(0, 6, txt="--- AI SUMMARY ---")
                pdf.multi_cell(0, 6, txt=ai_summary)

            report_path = "eda_report.pdf"
            pdf.output(report_path)

            with open(report_path, "rb") as f:
                st.download_button("Download Report", f, file_name="eda_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")

    st.success("EDA completed with local AI insights and suggestions.")
else:
    st.info("Upload a CSV to get started.")