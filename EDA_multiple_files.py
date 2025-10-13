import os
import io
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from openai import AzureOpenAI

# ---------------------- Azure OpenAI Setup ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")  # https://<resource>.openai.azure.com
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")  # Azure *deployment* name
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-10-21")  # GA version

client = None
try:
    if not OPENAI_API_KEY or not OPENAI_ENDPOINT or not OPENAI_DEPLOYMENT_NAME:
        raise ValueError("Missing env vars: OPENAI_API_KEY, OPENAI_ENDPOINT, OPENAI_DEPLOYMENT_NAME")
    if not OPENAI_ENDPOINT.startswith("https://"):
        raise ValueError("OPENAI_ENDPOINT must start with https://")
    client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version=OPENAI_API_VERSION,
    )
except Exception as e:
    st.error(f"Error initializing Azure OpenAI client: {e}")

# ---------------------- Page config ----------------------
st.set_page_config(page_title="Agentic EDA + Conversational Insights", layout="wide")
st.title("Agentic EDA Tool — Converse about Insights & Diagrams")
uploaded_files = st.file_uploader("Choose CSV files", type=["csv"], accept_multiple_files=True)

# Per-file chat threads and chat seeds
if "threads" not in st.session_state:
    st.session_state.threads = {}  # {file_key: [messages]}
if "chat_seed" not in st.session_state:
    st.session_state.chat_seed = {}  # {file_key: str}

# ---------------------- Helpers ----------------------
def compact_json(obj, limit_chars=7000):
    try:
        s = json.dumps(obj, ensure_ascii=False)
        return s[:limit_chars]
    except Exception:
        return str(obj)[:limit_chars]

def correlation_summary(corr_df: pd.DataFrame, top_k=5):
    try:
        pairs = []
        cols = list(corr_df.columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                val = float(corr_df.iloc[i, j])
                pairs.append((cols[i], cols[j], val))
        pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
        top_pos = [(a, b, round(v, 3)) for a, b, v in pairs if v > 0][:top_k]
        top_neg = [(a, b, round(v, 3)) for a, b, v in pairs if v < 0][:top_k]
        return top_pos, top_neg
    except Exception as e:
        st.error(f"Error summarizing correlations: {e}")
        return [], []

def iqr_outliers_count(s: pd.Series):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < low) | (s > high)).sum())

def ensure_thread(file_key):
    if file_key not in st.session_state.threads:
        st.session_state.threads[file_key] = []

def ai_call(messages, max_completion_tokens=16384):
    """Generic AI call with error handling."""
    if client is None:
        return "Azure OpenAI client is not initialized."
    try:
        resp = client.chat.completions.create(
            model=OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return ""

def build_eda_context(df, numeric_df, numeric_summary, missing_vals, outlier_summary, corr_ctx, cat_ctx):
    try:
        buf = io.StringIO()
        df.info(buf=buf)
        return {
            "shape": df.shape,
            "columns_head": df.columns.tolist()[:60],
            "data_info": buf.getvalue()[:3000],
            "missing_top": missing_vals.head(40).to_dict() if isinstance(missing_vals, pd.Series) else {},
            "numeric_summary_head": (numeric_summary.head(40).to_dict() if isinstance(numeric_summary, pd.DataFrame) and not numeric_summary.empty else {}),
            "outliers_top": sorted(outlier_summary.items(), key=lambda kv: kv[1], reverse=True)[:12] if outlier_summary else [],
            "corr_summary": corr_ctx or {},
            "categorical_summary": cat_ctx or {},
        }
    except Exception as e:
        st.error(f"Error building EDA context: {e}")
        return {}

def chat_about(file_key: str, user_text: str, eda_context: dict):
    """Multi-turn chat grounded in EDA context."""
    ensure_thread(file_key)
    thread = st.session_state.threads[file_key]

    messages = [{
        "role": "system",
        "content": ("You are a senior data analyst. Discuss EDA diagrams (preview, missing, numeric stats, correlations, boxplots, categorical) "
                    "with clear, practical guidance that ties to preprocessing, feature engineering, and modeling impact.")
    }]

    # Inject context once (first turn), and keep it in thread for grounding
    if not any(m.get("role") == "user" and m.get("tag") == "eda_context" for m in thread):
        try:
            ctx_text = compact_json(eda_context)
            messages.append({"role": "user", "content": f"EDA context:\n{ctx_text}", "tag": "eda_context"})
            thread.append({"role": "user", "content": f"EDA context:\n{ctx_text}", "tag": "eda_context"})
        except Exception as e:
            st.error(f"Error adding EDA context: {e}")

    # Add prior messages (excluding the stored eda_context marker)
    messages.extend({"role": m["role"], "content": m["content"]}
                    for m in thread if m.get("tag") != "eda_context")

    # Current user input
    messages.append({"role": "user", "content": user_text})

    reply = ai_call(messages, max_completion_tokens=16384)
    if reply:
        # Update thread
        thread.append({"role": "user", "content": user_text})
        thread.append({"role": "assistant", "content": reply})

        # Display this turn
        st.chat_message("user").write(user_text)
        st.chat_message("assistant").write(reply)

# ---------------------- Main Logic ----------------------
if not uploaded_files:
    st.info("Upload CSV files to get started.")
else:
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        with st.expander(f"📄 EDA & Conversation — {file_key}", expanded=True):
            # Read CSV
            try:
                df = pd.read_csv(uploaded_file, low_memory=False)
            except Exception as e:
                st.error(f"Error reading CSV '{file_key}': {e}")
                continue
            # ---- 1) Preview ----
            st.subheader("👀 Data Preview")
            try:
                st.dataframe(df.head(100))
            except Exception as e:
                st.error(f"Error displaying data preview: {e}")
            if st.button(f"Discuss Data Preview — {file_key}", key=f"btn_preview_{file_key}"):
                st.session_state.chat_seed[file_key] = "What does the preview suggest about types, targets, and immediate cleanup needs?"

            # ---- 2) Missing ----
            st.subheader("❓ Missing Values")
            try:
                missing_vals = df.isna().sum().sort_values(ascending=False)
                st.dataframe(missing_vals.rename("missing_count"))
            except Exception as e:
                st.error(f"Error computing missing values: {e}")
                missing_vals = pd.Series(dtype="int64")
            if st.button(f"Discuss Missing Values — {file_key}", key=f"btn_missing_{file_key}"):
                st.session_state.chat_seed[file_key] = (
                    "Which columns are most affected by missing values, and what imputation/handling strategies are best here?"
                )

            # ---- 3) Numeric summary ----
            st.subheader("🧮 Summary Statistics (Numeric)")
            try:
                numeric_df = df.select_dtypes(include=["number"])
                if numeric_df.empty:
                    st.warning("No numeric columns found.")
                    numeric_summary = pd.DataFrame()
                else:
                    numeric_summary = numeric_df.describe().T
                    st.dataframe(numeric_summary)
            except Exception as e:
                st.error(f"Error computing numeric summary: {e}")
                numeric_df, numeric_summary = pd.DataFrame(), pd.DataFrame()
            if st.button(f"Discuss Numeric Summary — {file_key}", key=f"btn_numeric_{file_key}"):
                st.session_state.chat_seed[file_key] = (
                    "Which numeric columns are skewed or high variance, and how should we scale/transform them?"
                )

            # ---- 4) Correlation heatmap ----
            st.subheader("🔗 Correlation Matrix")
            corr = pd.DataFrame()
            corr_ctx = {}
            if not numeric_df.empty:
                try:
                    # Choose top-variance columns to keep plot readable
                    var = numeric_df.var().sort_values(ascending=False)
                    cols = var.head(20).index.tolist()
                    corr = numeric_df[cols].corr(numeric_only=True)

                    fig, ax = plt.subplots(figsize=(min(0.5 * len(cols) + 4, 16), min(0.5 * len(cols) + 4, 16)))
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, square=True, cbar_kws={"shrink": 0.8})
                    ax.set_title("Correlation Heatmap (top-variance numeric features)")
                    plt.tight_layout()
                    st.pyplot(fig)

                    top_pos, top_neg = correlation_summary(corr, top_k=5)
                    corr_ctx = {"columns_plotted": cols, "top_positive_pairs": top_pos, "top_negative_pairs": top_neg}
                except Exception as e:
                    st.error(f"Error generating correlation heatmap: {e}")
                    corr = pd.DataFrame()
            if st.button(f"Discuss Correlations — {file_key}", key=f"btn_corr_{file_key}"):
                st.session_state.chat_seed[file_key] = (
                    "Which correlations matter most, and how should we handle multicollinearity and interaction terms?"
                )

            # ---- 5) Outliers & boxplots ----
            st.subheader("🚨 Outlier Detection (IQR Method)")
            outlier_summary = {}
            if not numeric_df.empty:
                try:
                    for col in numeric_df.columns:
                        outlier_summary[col] = iqr_outliers_count(numeric_df[col].dropna())

                    st.write("Outlier counts per column:")
                    st.json({k: int(v) for k, v in outlier_summary.items()})

                    top_cols = sorted(outlier_summary, key=outlier_summary.get, reverse=True)[:10]
                    for col in top_cols:
                        fig, ax = plt.subplots(figsize=(6, 2.8))
                        sns.boxplot(x=numeric_df[col], ax=ax, color="#6BAED6")
                        ax.set_title(f"Boxplot: {col} (outliers: {outlier_summary[col]})", fontsize=11)
                        plt.tight_layout()
                        st.pyplot(fig)

                        if st.button(f"Discuss boxplot: {col} — {file_key}", key=f"btn_box_{file_key}_{col}"):
                            st.session_state.chat_seed[file_key] = (
                                f"What do outliers in {col} imply? Should we cap, transform, or investigate sources—and why?"
                            )
                except Exception as e:
                    st.error(f"Error during outlier detection/plotting: {e}")

            # ---- 6) Categorical analysis ----
            st.subheader("🔤 Categorical Feature Analysis")
            cat_ctx = {}
            try:
                categorical_df = df.select_dtypes(include=["object", "category", "bool"])
                if categorical_df.empty:
                    st.info("No categorical columns found.")
                else:
                    for col in categorical_df.columns:
                        vc = df[col].value_counts(dropna=False).head(30)
                        st.write(f"Top categories for **{col}**:")
                        st.dataframe(vc.rename("count"))

                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax, color="#FD8D3C")
                        ax.set_title(f"Value counts: {col}")
                        ax.set_ylabel("count")
                        ax.set_xlabel(col)
                        for tick in ax.get_xticklabels():
                            tick.set_rotation(45)
                            tick.set_ha("right")
                        plt.tight_layout()
                        st.pyplot(fig)

                        cat_ctx[col] = {str(k): int(v) for k, v in vc.to_dict().items()}

                        if st.button(f"Discuss categorical: {col} — {file_key}", key=f"btn_cat_{file_key}_{col}"):
                            st.session_state.chat_seed[file_key] = (
                                f"In {col}, how should we handle imbalance or rare categories—grouping, one‑hot, target or frequency encoding?"
                            )
            except Exception as e:
                st.error(f"Error in categorical analysis: {e}")

            # ---- 7) AI Insight Cards (with discuss button) ----
            st.subheader("🧠 AI Insight Cards")
            try:
                insight_context = {
                    "Missing Values": missing_vals.to_dict() if isinstance(missing_vals, pd.Series) else {},
                    "Outliers": outlier_summary,
                    "Numeric Summary": (numeric_df.describe().to_dict() if not numeric_df.empty else {})
                }
                insight_messages = [
                    {"role": "system", "content": "You are a data analyst. Generate 3 concise, actionable insights."},
                    {"role": "user", "content": compact_json(insight_context)}
                ]
                insights_text = ai_call(insight_messages, max_completion_tokens=16384)
                insights = [line.strip("•- ").strip() for line in insights_text.split("\n") if line.strip()]
                for insight in insights:
                    st.info(insight)
                if st.button(f"Discuss these insights — {file_key}", key=f"btn_discuss_insights_{file_key}"):
                    st.session_state.chat_seed[file_key] = (
                        "Given the insight cards above, what should be our top data cleaning and feature engineering actions?"
                    )
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")

            # ---- 8) Data quality score ----
            st.subheader("🧪 Data Quality Score")
            try:
                total_cells = df.shape[0] * df.shape[1]
                missing_ratio = (missing_vals.sum() / total_cells) if total_cells > 0 else 0.0
                outlier_cells = int(sum(outlier_summary.values()))
                numeric_cells = int(numeric_df.shape[0] * numeric_df.shape[1]) if not numeric_df.empty else 1
                outlier_ratio = outlier_cells / numeric_cells
                quality_score = max(0.0, 100.0 - (missing_ratio * 60.0 + outlier_ratio * 40.0) * 100.0)
                st.metric("Estimated Data Quality Score", f"{quality_score:.2f}/100")
                if st.button(f"Discuss Data Quality — {file_key}", key=f"btn_dq_{file_key}"):
                    st.session_state.chat_seed[file_key] = (
                        "How do missing and outliers drive this quality score, and what should we fix first to improve it?"
                    )
            except Exception as e:
                st.error(f"Error computing quality score: {e}")

            # ---- Build EDA context for conversation ----
            eda_context = build_eda_context(
                df=df,
                numeric_df=numeric_df,
                numeric_summary=(numeric_summary if 'numeric_summary' in locals() else pd.DataFrame()),
                missing_vals=(missing_vals if 'missing_vals' in locals() else pd.Series(dtype="int64")),
                outlier_summary=outlier_summary,
                corr_ctx=corr_ctx,
                cat_ctx=cat_ctx
            )

            # ---------------- Conversational Panel ----------------
            st.markdown("---")
            st.subheader("💬 Converse about the EDA insights & diagrams")

            ensure_thread(file_key)

            # Show prior conversation
            for m in st.session_state.threads[file_key]:
                if m.get("tag") == "eda_context":
                    continue
                st.chat_message("assistant" if m["role"] == "assistant" else "user").write(m["content"])

            # Prefill from any “Discuss…” button above
            default_prompt = st.session_state.chat_seed.get(file_key, "")
            user_input = st.chat_input(
                "Ask about any diagram or insight…",
                key=f"chat_input_{file_key}",
                max_chars=2000,
                placeholder=(default_prompt or "What stands out in the EDA, and how should we preprocess/features?")
            )

            if user_input:
                # Clear one-time prefill
                try:
                    st.session_state.chat_seed[file_key] = ""
                except Exception as e:
                    st.error(f"Error clearing chat seed: {e}")
                try:
                    chat_about(file_key=file_key, user_text=user_input, eda_context=eda_context)
                except Exception as e:
                    st.error(f"Error calling OpenAI API: {e}")

            colA, colB = st.columns(2)
            with colA:
                if st.button(f"Summarize top actions — {file_key}", key=f"btn_sum_{file_key}"):
                    try:
                        chat_about(
                            file_key=file_key,
                            user_text=("Summarize the most important EDA takeaways and propose the top 5 preprocessing steps "
                                       "and top 5 feature engineering ideas, with brief rationale."),
                            eda_context=eda_context
                        )
                    except Exception as e:
                        st.error(f"Error calling OpenAI API: {e}")
            with colB:
                if st.button(f"Reset conversation — {file_key}", key=f"btn_reset_{file_key}"):
                    try:
                        st.session_state.threads[file_key] = []
                        st.success("Conversation reset for this file.")
                    except Exception as e:
                        st.error(f"Error resetting conversation: {e}")

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
