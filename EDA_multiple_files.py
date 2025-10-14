# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Agentic EDA (Single File)", layout="wide")

# -----------------------
# Session state keys
# -----------------------
CHAT_KEYS = {
    "Home": "chat_home",
    "Box Plots": "chat_box",
    "Correlation Matrix": "chat_corr",
}
DATA_KEYS = {
    "original_df": "original_df",
    "active_df": "active_df",
    "versions": "dataset_versions",  # list of dicts: {'name', 'df', 'diff'}
    "last_upload": "_last_upload",
}

# -----------------------
# Data I/O and state
# -----------------------
@st.cache_data(show_spinner=False)
def read_file(uploaded_file) -> pd.DataFrame:
    """Read CSV/Parquet/Excel into a DataFrame."""
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".parquet"):
        # Requires pyarrow or fastparquet installed
        return pd.read_parquet(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Use CSV/Parquet/Excel.")

def init_data(uploaded_file):
    """Initialize original and active data in session_state."""
    df = read_file(uploaded_file)
    st.session_state.setdefault(DATA_KEYS["original_df"], df.copy())
    st.session_state.setdefault(DATA_KEYS["active_df"], df.copy())
    st.session_state.setdefault(DATA_KEYS["versions"], [])

def get_active_df() -> pd.DataFrame:
    return st.session_state.get(DATA_KEYS["active_df"], pd.DataFrame())

def set_active_df(df, version_name=None, diff=None):
    """Set active df and optionally append a version entry."""
    st.session_state[DATA_KEYS["active_df"]] = df
    if version_name:
        versions = st.session_state.setdefault(DATA_KEYS["versions"], [])
        versions.append({"name": version_name, "df": df.copy(), "diff": diff})

def export_df(df: pd.DataFrame, fmt: str = "csv") -> bytes:
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif fmt == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)  # requires pyarrow
        return buf.getvalue()
    else:
        raise ValueError("Unsupported export format.")

# -----------------------
# Helpers: layout & charts
# -----------------------
def three_pane(title: str):
    """Create a consistent three-column layout."""
    st.title(title)
    left, mid, right = st.columns([1.1, 1.8, 1.1])
    return left, mid, right

def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

def render_boxplots(df: pd.DataFrame, groupby: str | None):
    nums = numeric_columns(df)
    if not nums:
        st.info("No numeric columns available for box plots.")
        return
    for col in nums:
        fig, ax = plt.subplots(figsize=(6, 4))
        if groupby and groupby in df.columns:
            sns.boxplot(x=df[groupby], y=df[col], ax=ax)
            ax.set_title(f"{col} by {groupby}")
            ax.set_xlabel(groupby)
            ax.set_ylabel(col)
        else:
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"{col} distribution")
            ax.set_ylabel(col)
        st.pyplot(fig, clear_figure=True)

def render_corr_matrix(df: pd.DataFrame, method: str = "pearson", mask_upper: bool = True, annot: bool = False):
    nums = numeric_columns(df)
    if len(nums) < 2:
        st.info("Need at least 2 numeric columns for correlation.")
        return
    corr = df[nums].corr(method=method)
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=annot, fmt=".2f", ax=ax)
    ax.set_title(f"Correlation Matrix ({method})")
    st.pyplot(fig, clear_figure=True)

# -----------------------
# Left pane: chat panel (stub)
# -----------------------
def llm_call(system_prompt: str, messages: list[dict]) -> str:
    """Stub for Azure OpenAI/GPT.nano call. Replace with your integration."""
    # You can build richer context by including chart controls & quick stats.
    return "🔎 (Stub) I would explain the chart based on the selected controls and current dataset."

def chat_panel(state_key: str, header: str = "🧠 Chart Understanding Chat", system_prompt: str = None):
    st.subheader(header)
    msgs = st.session_state.setdefault(state_key, [])
    # Render existing messages
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    # Input
    user_input = st.chat_input("Ask about this chart…")
    if user_input:
        msgs.append({"role": "user", "content": user_input})
        response = llm_call(system_prompt or "", msgs)
        msgs.append({"role": "assistant", "content": response})
        st.session_state[state_key] = msgs
    cols_btn = st.columns([1, 1])
    with cols_btn[0]:
        if st.button("Clear chat"):
            st.session_state[state_key] = []
            st.experimental_rerun()
    with cols_btn[1]:
        st.caption("Chat is scoped to this page (chart type).")

# -----------------------
# Right pane: data summary + deletion/export
# -----------------------
def data_summary_panel(df: pd.DataFrame):
    st.subheader("📚 Total Data Understanding")
    if df.empty:
        st.info("No data loaded yet.")
        return
    st.write(f"**Rows:** {df.shape[0]} **Columns:** {df.shape[1]}")
    st.markdown("**Column types:**")
    st.write(df.dtypes.astype(str))
    st.markdown("**Missing values (%):**")
    st.write((df.isna().mean() * 100).round(2))
    with st.expander("Preview (top 30 rows)"):
        st.dataframe(df.head(30), use_container_width=True)

def feature_delete_panel():
    st.subheader("🧹 Delete Features & Export New File")
    df = get_active_df()
    if df.empty:
        st.info("Upload data to manage features.")
        return

    drop_cols = st.multiselect("Select features to delete", options=list(df.columns), default=[])
    if drop_cols:
        st.caption(f"Will drop {len(drop_cols)} columns: {drop_cols}")

    version_name = st.text_input("Version name", value="features_dropped_v1")
    apply = st.button("Apply deletion to Active dataset")
    if apply:
        new_df = df.drop(columns=drop_cols) if drop_cols else df.copy()
        diff = {"dropped_columns": drop_cols}
        set_active_df(new_df, version_name=version_name, diff=diff)
        st.success(f"Applied. Active dataset updated → {new_df.shape[0]} rows × {new_df.shape[1]} columns")

    st.markdown("---")
    export_fmt = st.selectbox("Export format", ["csv", "parquet"])
    file_name = st.text_input("Export file name (without extension)", value="cleaned_dataset")
    if st.button("Generate downloadable file"):
        out = export_df(get_active_df(), fmt=export_fmt)
        st.download_button(
            label=f"Download {export_fmt.upper()}",
            data=out,
            file_name=f"{file_name}.{export_fmt}",
            mime="text/csv" if export_fmt == "csv" else "application/octet-stream",
            use_container_width=True,
        )

    st.markdown("---")
    if st.button("Undo last version change"):
        versions = st.session_state.get(DATA_KEYS["versions"], [])
        if versions:
            versions.pop()  # remove last record
            if versions:
                st.session_state[DATA_KEYS["active_df"]] = versions[-1]["df"].copy()
            else:
                st.session_state[DATA_KEYS["active_df"]] = st.session_state[DATA_KEYS["original_df"]].copy()
            st.warning("Reverted to previous active dataset.")
        else:
            st.info("No versions to undo.")

# -----------------------
# Sidebar: data source & navigation
# -----------------------
st.sidebar.title("📂 Data Source")
uploaded = st.sidebar.file_uploader("Upload CSV / Parquet / Excel", type=["csv", "parquet", "xlsx", "xls"])
if uploaded is not None and uploaded != st.session_state.get(DATA_KEYS["last_upload"]):
    st.session_state[DATA_KEYS["last_upload"]] = uploaded
    init_data(uploaded)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate pages", options=["Home", "Box Plots", "Correlation Matrix"])
st.sidebar.markdown("---")
st.sidebar.info("Left: chat for chart understanding\nMiddle: chart & controls\nRight: data understanding + feature deletion & export")

# -----------------------
# Pages
# -----------------------
if page == "Home":
    left, mid, right = three_pane("🏠 Home")
    df = get_active_df()
    with left:
        chat_panel(CHAT_KEYS["Home"], system_prompt="General EDA questions about the dataset.")
    with mid:
        st.subheader("Active Dataset Overview")
        if df.empty:
            st.info("Upload a dataset using the sidebar.")
        else:
            st.success(f"Active dataset: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(50), use_container_width=True)
    with right:
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()

elif page == "Box Plots":
    left, mid, right = three_pane("📦 Box Plots")
    df = get_active_df()
    with left:
        chat_panel(
            CHAT_KEYS["Box Plots"],
            system_prompt="Explain box plots, outliers (IQR), skewness, and group comparisons for selected columns."
        )
    with mid:
        st.subheader("🎛️ Controls")
        if df.empty:
            st.info("Upload data on Home page.")
        else:
            cats = categorical_columns(df)
            groupby = st.selectbox("Group by (optional)", options=["— None —"] + cats)
            groupby = None if groupby == "— None —" else groupby
            st.divider()
            render_boxplots(df, groupby=groupby)
    with right:
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()

elif page == "Correlation Matrix":
    left, mid, right = three_pane("🧩 Correlation Matrix")
    df = get_active_df()
    with left:
        chat_panel(
            CHAT_KEYS["Correlation Matrix"],
            system_prompt="Discuss correlations, multicollinearity (VIF), and feature selection implications."
        )
    with mid:
        st.subheader("🎛️ Controls")
        if df.empty:
            st.info("Upload data on Home page.")
        else:
            method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
            mask_upper = st.checkbox("Mask upper triangle", value=True)
            annot = st.checkbox("Show numeric annotations", value=False)
            st.divider()
            render_corr_matrix(df, method=method, mask_upper=mask_upper, annot=annot)
    with right:
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()