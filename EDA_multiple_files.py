import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
from openai import OpenAI
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

import os
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from openai import AzureOpenAI

# ---------------- Azure OpenAI Setup ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")  # e.g., https://your-resource-name.openai.azure.com
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")  # e.g., gpt-4o-mini (DEPLOYMENT NAME)
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-10-21")  # GA version; override if needed

client = None
try:
    if not OPENAI_API_KEY or not OPENAI_ENDPOINT or not OPENAI_DEPLOYMENT_NAME:
        raise ValueError(
            "Missing required env vars: OPENAI_API_KEY, OPENAI_ENDPOINT, OPENAI_DEPLOYMENT_NAME."
        )
    if not OPENAI_ENDPOINT.startswith("https://"):
        raise ValueError("OPENAI_ENDPOINT must start with https://")

    client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version=OPENAI_API_VERSION,
    )
except Exception as e:
    st.error(f"Error initializing Azure OpenAI client: {e}")

# ---------------- Streamlit UI ----------------
st.title("Agentic EDA Tool with AI Insights & Model Training")
st.write("Upload one or more CSV files to begin automated exploratory data analysis and interact with AI for deeper insights.")

uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

# ---------------- Helper ----------------
def _safe_numeric_describe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        numeric_df = df.select_dtypes(include=["number"])
        return numeric_df, numeric_df.describe()
    except Exception as e:
        st.error(f"Error during numeric summary: {e}")
        return pd.DataFrame(), pd.DataFrame()

def _ai_call(messages, max_completion_tokens=16384) -> str:
    """Minimal wrapper for AI calls with error handling."""
    if client is None:
        return "Azure OpenAI client is not initialized."
    try:
        resp = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,  # IMPORTANT: deployment name, not base model
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return ""

# ---------------- Main Logic ----------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"EDA & Modeling for {uploaded_file.name}", expanded=True):
            # Robust CSV read
            try:
                df = pd.read_csv(uploaded_file, low_memory=False)
            except Exception as e:
                st.error(f"Error reading CSV '{uploaded_file.name}': {e}")
                continue

            # --- BASIC EDA ---
            st.subheader("Data Preview")
            try:
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error displaying data preview: {e}")

            st.subheader("Basic Information")
            try:
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
            except Exception as e:
                st.error(f"Error retrieving basic info: {e}")

            st.subheader("Missing Values")
            try:
                missing_vals = df.isnull().sum()
                st.write(missing_vals)
            except Exception as e:
                st.error(f"Error computing missing values: {e}")
                missing_vals = pd.Series(dtype="int64")

            # --- NUMERIC SUMMARY ---
            st.subheader("Summary Statistics (Numeric Columns Only)")
            numeric_df, numeric_summary = _safe_numeric_describe(df)
            try:
                if numeric_df.empty:
                    st.warning("No numeric columns found.")
                else:
                    st.write(numeric_summary)
            except Exception as e:
                st.error(f"Error showing numeric summary: {e}")

            # --- CORRELATION MATRIX ---
            st.subheader("Correlation Matrix (Numeric Columns Only)")
            if not numeric_df.empty:
                try:
                    corr = numeric_df.corr(numeric_only=True)
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating correlation heatmap: {e}")

            # --- OUTLIER DETECTION ---
            st.subheader("Outlier Detection (IQR Method)")
            outlier_summary = {}
            if not numeric_df.empty:
                try:
                    for col in numeric_df.columns:
                        Q1 = numeric_df[col].quantile(0.25)
                        Q3 = numeric_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = numeric_df[(numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)]
                        outlier_summary[col] = len(outliers)

                        fig, ax = plt.subplots()
                        sns.boxplot(x=numeric_df[col], ax=ax)
                        ax.set_title(f"Boxplot: {col} (outliers: {outlier_summary[col]})")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error during outlier detection: {e}")
            st.write("Outlier counts per column:", outlier_summary)

            # --- DATA QUALITY SCORE ---
            st.subheader("Data Quality Score")
            try:
                total_cells = df.size
                missing_ratio = (missing_vals.sum() / total_cells) if total_cells > 0 else 0.0
                outlier_ratio = (sum(outlier_summary.values()) / numeric_df.size) if not numeric_df.empty else 0.0
                quality_score = max(0.0, 100.0 - (missing_ratio * 50.0 + outlier_ratio * 50.0))
                st.write(f"Estimated Data Quality Score: {quality_score:.2f}/100")
            except Exception as e:
                st.error(f"Error computing quality score: {e}")

            # --- CATEGORICAL ANALYSIS ---
            st.subheader("Categorical Feature Analysis")
            try:
                categorical_df = df.select_dtypes(include=["object", "category"])
                if categorical_df.empty:
                    st.info("No categorical columns found.")
                else:
                    for col in categorical_df.columns:
                        st.write(f"Value Counts for {col}:")
                        st.write(df[col].value_counts())

                        fig, ax = plt.subplots()
                        # Limit to top 30 categories for readability
                        plot_df = df[col].value_counts().head(30)
                        sns.barplot(x=plot_df.index.astype(str), y=plot_df.values, ax=ax)
                        ax.set_xlabel(col)
                        ax.set_ylabel("count")
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in categorical analysis: {e}")

            # --- AI CHART EXPLANATION ---
            st.subheader("AI Chart Explanation")
            chart_question = st.text_input(
                f"Ask AI to explain a chart or pattern in {uploaded_file.name}",
                key=f"chart_q_{uploaded_file.name}"
            )
            if st.button(f"Explain Chart for {uploaded_file.name}"):
                try:
                    chart_context = (
                        f"Numeric Summary: {numeric_summary.to_dict() if not numeric_summary.empty else {}} "
                        f"\nOutlier Summary: {outlier_summary}"
                    )
                    messages = [
                        {"role": "system", "content": "You are a data analyst. Explain the chart and patterns in simple terms."},
                        {"role": "user", "content": chart_context},
                        {"role": "user", "content": chart_question or "Explain key patterns."}
                    ]
                    reply = _ai_call(messages, max_completion_tokens=16384)
                    st.write("### AI Chart Explanation")
                    st.write(reply)
                except Exception as e:
                    st.error(f"Error calling OpenAI API: {e}")

            # --- INSIGHT CARDS ---
            st.subheader("AI Insight Cards")
            try:
                insight_context = (
                    f"Missing Values: {missing_vals.to_dict()} "
                    f"\nOutliers: {outlier_summary} "
                    f"\nNumeric Summary: {numeric_summary.to_dict() if not numeric_summary.empty else {}}"
                )
                messages = [
                    {"role": "system", "content": "You are a data analyst. Generate 3 key insights from the dataset."},
                    {"role": "user", "content": insight_context}
                ]
                insights_text = _ai_call(messages, max_completion_tokens=16384)
                insights = [line for line in insights_text.split("\n") if line.strip()]
                for insight in insights:
                    st.info(insight)
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")

            # --- AUTO SUGGESTIONS ---
            st.subheader("Auto Suggestions for Fixes")
            try:
                suggestion_context = f"Missing Values: {missing_vals.to_dict()}\nOutliers: {outlier_summary}"
                messages = [
                    {"role": "system", "content": "You are a data scientist. Suggest preprocessing steps to clean the data."},
                    {"role": "user", "content": suggestion_context}
                ]
                suggestions_text = _ai_call(messages, max_tokens=500)
                suggestions = [line for line in suggestions_text.split("\n") if line.strip()]
                for suggestion in suggestions:
                    st.warning(suggestion)
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")


            # --- MODEL TRAINING ---
            st.subheader("Model Training: Gradient Boosting")
            target_col = st.selectbox("Select Target Column", df.columns, key=f"target_{uploaded_file.name}")

            if target_col:
                features = df.drop(columns=[target_col])
                target = df[target_col]

                for col in features.select_dtypes(include=['object', 'category']).columns:
                    features[col] = LabelEncoder().fit_transform(features[col].astype(str))

                is_classification = target.nunique() <= 10 and target.dtype == 'object'
                if is_classification:
                    target = LabelEncoder().fit_transform(target.astype(str))

                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

                model_type = "Classifier" if is_classification else "Regressor"
                st.write(f"Training Gradient Boosting {model_type}...")
                model = GradientBoostingClassifier() if is_classification else GradientBoostingRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                tab1, tab2, tab3 = st.tabs(["📊 Evaluation", "📈 Feature Importance", "✅ Business Summary"])

                with tab1:
                    st.subheader("Model Evaluation")
                    if is_classification:
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.write("Classification Report:")
                        st.json(report)
                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        st.write(f"Mean Squared Error: {mse:.2f}")
                        st.write(f"R² Score: {r2:.2f}")

                with tab2:
                    st.subheader("Feature Importance")
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': features.columns,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    st.write(importance_df)
                    fig, ax = plt.subplots()
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                    st.pyplot(fig)

                with tab3:
                    st.subheader("Business-Friendly Accuracy Summary")
                    if is_classification:
                        accuracy = model.score(X_test, y_test)
                        accuracy_percent = accuracy * 100
                        st.metric(label="Model Accuracy", value=f"{accuracy_percent:.2f}%")
                    else:
                        r2 = r2_score(y_test, y_pred)
                        accuracy_percent = r2 * 100
                        st.metric(label="Model R² Score (as Accuracy)", value=f"{accuracy_percent:.2f}%")

                    if accuracy_percent >= 90:
                        summary = "✅ Excellent model performance. Highly reliable for business decisions."
                    elif accuracy_percent >= 75:
                        summary = "🟢 Good model performance. Suitable for most business use cases."
                    elif accuracy_percent >= 60:
                        summary = "🟡 Moderate performance. May need further tuning or more data."
                    else:
                        summary = "🔴 Low performance. Not recommended for critical decisions without improvements."

                    st.info(summary)

            # --- REPORT GENERATION ---
            st.subheader("Downloadable Report")
            if st.button(f"Generate PDF Report for {uploaded_file.name}"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"EDA Report - {uploaded_file.name}", ln=True, align='C')
                pdf.multi_cell(0, 10, txt=f"Missing Values:\n{missing_vals.to_string()}\n\nOutliers:\n{outlier_summary}\n\nData Quality Score: {quality_score:.2f}/100")
                report_path = f"{uploaded_file.name}_eda_report.pdf"
                pdf.output(report_path)
                with open(report_path, "rb") as f:
                    st.download_button("Download Report", f, file_name=report_path)

            st.success(f"EDA and model training completed for {uploaded_file.name}.")
