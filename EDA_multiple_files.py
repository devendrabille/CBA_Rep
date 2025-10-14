
# app.py
import os
import io
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from openai import AzureOpenAI

# ------------------------ Azure OpenAI Setup ------------------------
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

# ------------------------ Page & Session ------------------------
st.set_page_config(page_title="Agentic EDA — Charts (Left) & Insights (Right)", layout="wide")
st.title("Agentic EDA Tool — Discuss Charts (Left) and Insights/Feature Engineering (Right)")

if "chart_threads" not in st.session_state:
    st.session_state.chart_threads = {}   # per-file, chart discussions
if "insight_threads" not in st.session_state:
    st.session_state.insight_threads = {} # per-file, insights discussions
if "seeds" not in st.session_state:
    st.session_state.seeds = {}           # quick prefill per file for left/right chats

# ------------------------ Helpers ------------------------
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

def ai_call(messages, max_completion_tokens=16384):
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

def ensure_threads(file_key):
    st.session_state.chart_threads.setdefault(file_key, [])
    st.session_state.insight_threads.setdefault(file_key, [])

def build_chart_context(df, numeric_df, missing_vals, corr_ctx, outliers_ctx, cat_ctx):
    try:
        buf = io.StringIO()
        df.info(buf=buf)
        return {
            "shape": df.shape,
            "columns_head": df.columns.tolist()[:60],
            "data_info": buf.getvalue()[:2500],
            "charts": {
                "missing_top": missing_vals.head(30).to_dict() if isinstance(missing_vals, pd.Series) else {},
                "corr_summary": corr_ctx or {},
                "outliers_top": outliers_ctx or [],
                "categorical_summary": cat_ctx or {},
                "numeric_cols_count": int(numeric_df.shape[1]) if isinstance(numeric_df, pd.DataFrame) else 0,
            }
        }
    except Exception as e:
        st.error(f"Error building chart context: {e}")
        return {}

def build_insight_context(df, numeric_df, numeric_summary, missing_vals, outlier_summary, dq_score, insight_cards, feat_ideas):
    try:
        return {
            "shape": df.shape,
            "missing_cols": int((missing_vals > 0).sum()) if isinstance(missing_vals, pd.Series) else 0,
            "numeric_summary_head": (numeric_summary.head(30).to_dict() if isinstance(numeric_summary, pd.DataFrame) and not numeric_summary.empty else {}),
            "outliers_top": sorted(outlier_summary.items(), key=lambda kv: kv[1], reverse=True)[:12] if outlier_summary else [],
            "data_quality_score": dq_score,
            "insight_cards": insight_cards or [],
            "feature_engineering_ideas": feat_ideas or [],
        }
    except Exception as e:
        st.error(f"Error building insight context: {e}")
        return {}

def chat_with(thread_list, user_text, system_prompt, context_text):
    # Inject context on first turn
    messages = [{"role": "Data Analyst", "content": system_prompt}]
    if not any(m.get("role") == "user" and m.get("tag") == "context" for m in thread_list):
        messages.append({"role": "user", "content": f"Context:\n{context_text}", "tag": "context"})
        thread_list.append({"role": "user", "content": f"Context:\n{context_text}", "tag": "context"})
    # Prior history
    messages.extend({"role": m["role"], "content": m["content"]} for m in thread_list if m.get("tag") != "context")
    # Current user input
    messages.append({"role": "user", "content": user_text})
    reply = ai_call(messages, max_completion_tokens=16384)
    if reply:
        thread_list.append({"role": "user", "content": user_text})
        thread_list.append({"role": "assistant", "content": reply})
        st.chat_message("user").write(user_text)
        st.chat_message("assistant").write(reply)

# ------------------------ Upload & EDA ------------------------
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more CSV files to start EDA.")
    st.stop()

for uploaded_file in uploaded_files:
    file_key = uploaded_file.name
    ensure_threads(file_key)

    with st.expander(f"📄 EDA for {file_key}", expanded=True):
        # Read CSV robustly
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
        except Exception as e:
            st.error(f"Error reading CSV '{file_key}': {e}")
            continue

        # 1) Preview
        st.subheader("👀 Data Preview")
        try:
            st.dataframe(df.head(100))
        except Exception as e:
            st.error(f"Error displaying preview: {e}")

        # 2) Missing
        st.subheader("❓ Missing Values")
        try:
            missing_vals = df.isna().sum().sort_values(ascending=False)
            st.dataframe(missing_vals.rename("missing_count"))
        except Exception as e:
            st.error(f"Error computing missing values: {e}")
            missing_vals = pd.Series(dtype="int64")

        # 3) Numeric summary
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

        # 4) Correlation heatmap
        st.subheader("🔗 Correlation Matrix")
        corr_ctx = {}
        try:
            if not numeric_df.empty and numeric_df.shape[1] >= 2:
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
            else:
                st.info("Not enough numeric columns for correlation heatmap.")
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {e}")

        # 5) Outliers & boxplots
        st.subheader("🚨 Outlier Detection (IQR Method)")
        outliers_ctx = []
        outlier_summary = {}
        try:
            if not numeric_df.empty:
                for col in numeric_df.columns:
                    cnt = iqr_outliers_count(numeric_df[col].dropna())
                    outlier_summary[col] = cnt
                st.write("Outlier counts per numeric column:")
                st.json({k: int(v) for k, v in outlier_summary.items()})
                # plot top 10
                top_cols = sorted(outlier_summary, key=outlier_summary.get, reverse=True)[:10]
                outliers_ctx = [(c, int(outlier_summary[c])) for c in top_cols]
                for col in top_cols:
                    fig, ax = plt.subplots(figsize=(6, 2.8))
                    sns.boxplot(x=numeric_df[col], ax=ax, color="#6BAED6")
                    ax.set_title(f"Boxplot: {col} (outliers: {outlier_summary[col]})", fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("No numeric columns for outlier analysis.")
        except Exception as e:
            st.error(f"Error during outlier detection/plotting: {e}")

        # 6) Categorical analysis
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
        except Exception as e:
            st.error(f"Error in categorical analysis: {e}")

        # 7) Data Quality Score
        st.subheader("🧪 Data Quality Score")
        dq_score = None
        try:
            total_cells = df.shape[0] * df.shape[1]
            missing_ratio = (missing_vals.sum() / total_cells) if total_cells > 0 else 0.0
            outlier_cells = int(sum(outlier_summary.values()))
            numeric_cells = int(numeric_df.shape[0] * numeric_df.shape[1]) if not numeric_df.empty else 1
            outlier_ratio = outlier_cells / numeric_cells
            dq_score = max(0.0, 100.0 - (missing_ratio * 60.0 + outlier_ratio * 40.0) * 100.0)
            st.metric("Estimated Data Quality Score", f"{dq_score:.2f}/100")
        except Exception as e:
            st.error(f"Error computing data quality score: {e}")

        # 8) AI Insight Cards & Feature Ideas (used by RIGHT chat context)
        st.subheader("🧠 AI Insight Cards")
        insight_cards = []
        feat_ideas = []
        try:
            insight_context = {
                "Missing Values": missing_vals.to_dict() if isinstance(missing_vals, pd.Series) else {},
                "Outliers": outlier_summary,
                "Numeric Summary": (numeric_df.describe().to_dict() if not numeric_df.empty else {})
            }
            msgs = [
                {"role": "Data Analyst", "content": "You are a data analyst. Generate 3 concise, actionable insights."},
                {"role": "user", "content": compact_json(insight_context)}
            ]
            txt = ai_call(msgs, max_completion_tokens=16384)
            insight_cards = [line.strip("•- ").strip() for line in txt.split("\n") if line.strip()]
            for card in insight_cards:
                st.info(card)
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")

        st.subheader("🧪 Feature Engineering Ideas")
        try:
            fe_msgs = [
                {"role": "Data Scientist", "content": "You are a feature engineering expert. Provide 5 specific ideas with short rationale."},
                {"role": "user", "content": compact_json(insight_context)}
            ]
            fe_txt = ai_call(fe_msgs, max_completion_tokens=16384)
            feat_ideas = [line.strip("•- ").strip() for line in fe_txt.split("\n") if line.strip()]
            for idea in feat_ideas:
                st.success(idea)
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")

    # ------------------------ DESIGN: Two-pane Conversation ------------------------
    st.markdown("### 💬 Discuss your EDA (Left: Charts) vs (Right: Insights & Feature Engineering)")

    # Build contexts for each pane
    chart_context = build_chart_context(df, numeric_df, missing_vals, corr_ctx, outliers_ctx, cat_ctx)
    insight_context_full = build_insight_context(df, numeric_df, numeric_summary, missing_vals, outlier_summary, dq_score, insight_cards, feat_ideas)

    # Prefill seeds from previous clicks (optional)
    seeds = st.session_state.seeds.get(file_key, {"left": "", "right": ""})

    # Two equally sized columns
    left_col, right_col = st.columns(2)

    # ---------- LEFT: Discuss Charts ----------
    with left_col:
        st.header("📊 Discuss Charts")
        st.caption("Talk about the EDA diagrams: preview table, missing values, correlation heatmap, boxplots, categorical counts.")
        # Show previous chart discussion
        for m in st.session_state.chart_threads[file_key]:
            if m.get("tag") == "context":  # skip context echo
                continue
            st.chat_message("assistant" if m["role"] == "Data Analyst" else "user").write(m["content"])
        # Chat input for charts (use placeholder as keyword ONLY)
        user_left = st.chat_input(
            placeholder=(seeds.get("left") or "Ask about any chart (e.g., correlations, outliers, categorical imbalance)…"),
            key=f"chat_charts_{file_key}",
            max_chars=2000,
        )
        if user_left:
            # clear seed
            st.session_state.seeds[file_key] = {"left": "", "right": seeds.get("right", "")}
            try:
                ctx_text = compact_json(chart_context)
                system_prompt = (
                    "You are a senior data analyst. Discuss the charts clearly and practically. "
                    "Highlight notable patterns, risks, and actions for preprocessing & modeling."
                )
                chat_with(st.session_state.chart_threads[file_key], user_left, system_prompt, ctx_text)
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")

        # Quick seed buttons
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(f"What stands out in correlations? — {file_key}", key=f"seed_corr_{file_key}"):
                st.session_state.seeds[file_key] = {"left": "Which relationships in the heatmap matter most and how to handle multicollinearity?", "right": seeds.get("right", "")}
                st.rerun()
        with c2:
            if st.button(f"Outliers handling? — {file_key}", key=f"seed_out_{file_key}"):
                st.session_state.seeds[file_key] = {"left": "How should we treat strong outliers—capping, robust scaling, or transformations?", "right": seeds.get("right", "")}
                st.rerun()
        with c3:
            if st.button(f"Categorical imbalance? — {file_key}", key=f"seed_cat_{file_key}"):
                st.session_state.seeds[file_key] = {"left": "What’s the best encoding given rare categories and imbalance?", "right": seeds.get("right", "")}
                st.rerun()

    # ---------- RIGHT: Discuss Insights & Feature Engineering ----------
    with right_col:
        st.header("🧠 Discuss Insights & Feature Engineering")
        st.caption("Ask about AI insight cards, data quality score, and concrete feature engineering strategies.")
        # Show previous insights discussion
        for m in st.session_state.insight_threads[file_key]:
            if m.get("tag") == "context":  # skip context echo
                continue
            st.chat_message("assistant" if m["role"] == "assistant" else "user").write(m["content"])
        # Chat input for insights
        user_right = st.chat_input(
            placeholder=(seeds.get("right") or "Ask about cleaning priorities, key insights, and feature ideas…"),
            key=f"chat_insights_{file_key}",
            max_chars=2000,
        )
        if user_right:
            # clear seed
            st.session_state.seeds[file_key] = {"left": seeds.get("left", ""), "right": ""}
            try:
                ctx_text = compact_json(insight_context_full)
                system_prompt = (
                    "You are a principal data scientist. Use the insights, quality score, and EDA signals to propose "
                    "cleaning priorities and feature engineering plans with clear rationales."
                )
                chat_with(st.session_state.insight_threads[file_key], user_right, system_prompt, ctx_text)
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")

        # Quick seed buttons
        r1, r2, r3 = st.columns(3)
        with r1:
            if st.button(f"Top 5 cleaning steps — {file_key}", key=f"seed_clean_{file_key}"):
                st.session_state.seeds[file_key] = {"left": seeds.get("left", ""), "right": "What are the top 5 preprocessing steps to improve data quality (with reasons)?"}
                st.rerun()
        with r2:
            if st.button(f"Top 5 feature ideas — {file_key}", key=f"seed_feat_{file_key}"):
                st.session_state.seeds[file_key] = {"left": seeds.get("left", ""), "right": "Suggest 5 feature engineering ideas tailored to these signals, with brief rationale."}
                st.rerun()
        with r3:
            if st.button(f"Modeling implications — {file_key}", key=f"seed_model_{file_key}"):
                st.session_state.seeds[file_key] = {"left": seeds.get("left", ""), "right": "How do these insights affect model choice, regularization, and evaluation strategy?"}
                st.rerun()
