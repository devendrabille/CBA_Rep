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
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# --- Azure OpenAI Client Setup ---
client = AzureOpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    api_version="2023-07-01-preview",
    azure_endpoint=st.secrets["OPENAI_ENDPOINT"]
)

st.title("Agentic EDA Tool with AI Insights & Model Training")
st.write("Upload one or more CSV files to begin automated exploratory data analysis and interact with AI for deeper insights.")

uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"EDA & Modeling for {uploaded_file.name}", expanded=True):
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
            chart_question = st.text_input(f"Ask AI to explain a chart or pattern in {uploaded_file.name}", key=f"chart_q_{uploaded_file.name}")
            if st.button(f"Explain Chart for {uploaded_file.name}"):
                chart_context = f"Numeric Summary: {numeric_df.describe().to_dict()}\nOutlier Summary: {outlier_summary}"
                messages = [
                    {"role": "system", "content": "An AI Assist to business and data analytics teams who would ask queries about the data as part of EDA and to provide excellent results and interpret the good patterns among data and good enough to suggest the fields for feature engineering. Not to answer any invalid questions and to prompt \"Try again with a valid data related context\"\n## To Avoid Harmful Content\n- You must not generate content that may be harmful to someone physically or emotionally even if a user requests or creates a condition to rationalize that harmful content.\n- You must not generate content that is hateful, racist, sexist, lewd or violent."},
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
                {"role": "system", "content": "An AI Assist to business and data analytics teams who would ask queries about the data as part of EDA and to provide excellent results and interpret the good patterns among data and good enough to suggest the fields for feature engineering. Not to answer any invalid questions and to prompt \"Try again with a valid data related context\"\n## To Avoid Harmful Content\n- You must not generate content that may be harmful to someone physically or emotionally even if a user requests or creates a condition to rationalize that harmful content.\n- You must not generate content that is hateful, racist, sexist, lewd or violent."},
                {"role": "user", "content": insight_context}
            ]
            try:
                response = client.chat.completions.create(
                    deployment_id=st.secrets["OPENAI_DEPLOYMENT"],
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
                    deployment_id=st.secrets["OPENAI_DEPLOYMENT"],
                    messages=messages,
                    max_completion_tokens=16384,
                    model=deployment
                )
                suggestions = response.choices[0].message.content.split("\n")
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
