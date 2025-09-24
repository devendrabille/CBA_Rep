import streamlit as st
import pandas as pd
import os
from openai import AzureOpenAI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

deployment = "simplegptnano"

# --- Azure OpenAI Client Setup ---
client = AzureOpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    api_version="2024-12-01-preview",
    azure_endpoint=st.secrets["OPENAI_ENDPOINT"]
)

# --- Streamlit UI ---
st.title("Agentic EDA Tool with AI Insights & Model Training")
st.write("Upload one or more CSV files to begin automated exploratory data analysis and interact with AI for deeper insights.")

uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"EDA & Modeling for {uploaded_file.name}", expanded=True):
            df = pd.read_csv(uploaded_file)

            # --- INSIGHT CARDS ---
            st.subheader("AI Insight Cards")
            missing_vals = df.isnull().sum()
            outlier_summary = df.describe().to_string()
            numeric_df = df.select_dtypes(include='number')

            insight_context = f"Missing Values: {missing_vals.to_dict()}\nOutliers: {outlier_summary}\nNumeric Summary: {numeric_df.describe().to_dict()}"
            messages = [
                {"role": "system", "content": "You are a data analyst. Generate 3 key insights from the dataset."},
                {"role": "user", "content": insight_context}
            ]
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    max_completion_tokens=16384,
                    model=deployment
                )
                insights = response.choices[0].message.content.split("\n")
                for insight in insights:
                    st.info(insight)
            except Exception as e:
                st.error(f"Error calling Azure OpenAI API: {e}")

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
                    max_tokens=300
                )
                suggestions = response.choices[0].message.content.split("\n")
                for suggestion in suggestions:
                    st.warning(suggestion)
            except Exception as e:
                st.error(f"Error calling Azure OpenAI API: {e}")

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
                pdf.multi_cell(0, 10, txt=f"Missing Values:\n{missing_vals.to_string()}\n\nOutliers:\n{outlier_summary}")
                report_path = f"{uploaded_file.name}_eda_report.pdf"
                pdf.output(report_path)
                with open(report_path, "rb") as f:
                    st.download_button("Download Report", f, file_name=report_path)

            st.success(f"EDA and model training completed for {uploaded_file.name}.")
