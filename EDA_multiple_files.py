# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from datetime import datetime

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Agentic EDA (Batch Across Multiple Files)", layout="wide")

# -----------------------
# Session state keys
# -----------------------
SS = {
    "datasets": "datasets",            # dict: name -> {"original": df, "active": df, "versions": []}
    "active_name": "active_name",      # str: current dataset name
}

CHAT_KEYS = {
    "Home": "chat_home",
    "Multi-EDA Summary": "chat_summary",
    "Box Plots": "chat_box",
    "Correlation Matrix": "chat_corr",
    "Bar Charts": "chat_bar",
    "Line Charts": "chat_line",
    "Scatter Plots": "chat_scatter",
    "Bubble Charts": "chat_bubble",
}

# -----------------------
# Data I/O
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

def _ensure_unique_name(existing, base):
    """Ensure dataset name uniqueness by adding suffixes when needed."""
    if base not in existing:
        return base
    idx = 1
    while f"{base} ({idx})" in existing:
        idx += 1
    return f"{base} ({idx})"

def ingest_files(files):
    """Load multiple files into session_state datasets."""
    if SS["datasets"] not in st.session_state:
        st.session_state[SS["datasets"]] = {}
    datasets = st.session_state[SS["datasets"]]

    added_names = []
    for f in files:
        try:
            df = read_file(f)
            base = f.name
            name = _ensure_unique_name(datasets.keys(), base)
            datasets[name] = {
                "original": df.copy(),
                "active": df.copy(),
                "versions": []  # list of dicts: {'name', 'df', 'diff'}
            }
            added_names.append(name)
        except Exception as e:
            st.sidebar.error(f"Failed to read {f.name}: {e}")

    # Set active dataset if none selected
    if added_names and not st.session_state.get(SS["active_name"]):
        st.session_state[SS["active_name"]] = added_names[0]

def get_active_df() -> pd.DataFrame:
    datasets = st.session_state.get(SS["datasets"], {})
    name = st.session_state.get(SS["active_name"])
    if not name or name not in datasets:
        return pd.DataFrame()
    return datasets[name]["active"]

def set_active_df(new_df: pd.DataFrame, version_name=None, diff=None):
    datasets = st.session_state.get(SS["datasets"], {})
    name = st.session_state.get(SS["active_name"])
    if not name or name not in datasets:
        return
    datasets[name]["active"] = new_df
    if version_name:
        datasets[name]["versions"].append({"name": version_name, "df": new_df.copy(), "diff": diff})

def set_all_active_df(drop_cols, base_version_name: str):
    """Apply feature deletion across all datasets (only dropping columns that exist)."""
    datasets = st.session_state.get(SS["datasets"], {})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, item in datasets.items():
        df = item["active"]
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        new_df = df.drop(columns=cols_to_drop) if cols_to_drop else df.copy()
        item["active"] = new_df
        item["versions"].append({
            "name": f"{base_version_name}__{name}__{ts}",
            "df": new_df.copy(),
            "diff": {"dropped_columns": cols_to_drop}
        })

def undo_last_version(active_only=True):
    datasets = st.session_state.get(SS["datasets"], {})
    if active_only:
        name = st.session_state.get(SS["active_name"])
        if not name or name not in datasets:
            return "No active dataset."
        versions = datasets[name]["versions"]
        if not versions:
            datasets[name]["active"] = datasets[name]["original"].copy()
            return "No versions to undo. Reverted to original."
        versions.pop()
        if versions:
            datasets[name]["active"] = versions[-1]["df"].copy()
            return "Reverted to previous version (active dataset)."
        else:
            datasets[name]["active"] = datasets[name]["original"].copy()
            return "Reverted to original (active dataset)."
    else:
        msg = []
        for name, item in datasets.items():
            versions = item["versions"]
            if not versions:
                item["active"] = item["original"].copy()
                msg.append(f"{name}: reverted to original")
            else:
                versions.pop()
                if versions:
                    item["active"] = versions[-1]["df"].copy()
                    msg.append(f"{name}: reverted to previous version")
                else:
                    item["active"] = item["original"].copy()
                    msg.append(f"{name}: reverted to original")
        return " ; ".join(msg)

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
# Helpers: layout & utilities
# -----------------------
def three_pane(title: str):
    st.title(title)
    left, mid, right = st.columns([1.1, 1.8, 1.1])
    return left, mid, right

def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

def datetime_columns(df: pd.DataFrame):
    dt_cols = []
    # include native datetime dtypes
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            dt_cols.append(col)
    # attempt to parse object columns
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            pd.to_datetime(df[col], errors="raise")
            dt_cols.append(col)
        except Exception:
            pass
    return list(dict.fromkeys(dt_cols))  # de-duplicate preserving order

def ensure_datetime(series: pd.Series):
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    try:
        return pd.to_datetime(series)
    except Exception:
        return series  # fallback; caller should handle

def iterate_datasets(render_fn, **kwargs):
    """Render a chart/function for ALL datasets as tabs."""
    datasets = st.session_state.get(SS["datasets"], {})
    if not datasets:
        st.info("Upload datasets to run batch EDA.")
        return
    names = sorted(datasets.keys())
    tabs = st.tabs(names)
    for tab, name in zip(tabs, names):
        with tab:
            df = datasets[name]["active"]
            st.markdown(f"**Dataset:** `{name}`  |  Shape: {df.shape[0]} × {df.shape[1]}")
            render_fn(df, dataset_name=name, **kwargs)

# -----------------------
# Left pane: chat (stubbed LLM)
# -----------------------
def llm_call(system_prompt: str, messages: list) -> str:
    # Replace with Azure OpenAI/GPT.nano integration
    return "🔎 (Stub) I would explain the chart based on selected controls and the current dataset(s)."

def chat_panel(state_key: str, header: str, system_prompt: str = None):
    st.subheader(header)
    msgs = st.session_state.setdefault(state_key, [])
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
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
# Right pane: data summary + deletion/export (Active or All)
# -----------------------
def data_summary_panel(df: pd.DataFrame):
    st.subheader("📚 Total Data Understanding (Active dataset)")
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
    datasets = st.session_state.get(SS["datasets"], {})
    active_df = get_active_df()
    if not datasets or active_df.empty:
        st.info("Upload data to manage features.")
        return

    scope = st.radio("Apply deletion to", options=["Active dataset", "All datasets"], horizontal=True)
    df_for_options = active_df if scope == "Active dataset" else active_df  # use active to list columns
    drop_cols = st.multiselect("Select features to delete", options=list(df_for_options.columns), default=[])
    if drop_cols:
        st.caption(f"Will drop {len(drop_cols)} column(s): {drop_cols}")

    version_name = st.text_input("Version name base", value=f"features_dropped_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    apply = st.button("Apply deletion")
    if apply:
        if scope == "Active dataset":
            new_df = active_df.drop(columns=drop_cols) if drop_cols else active_df.copy()
            diff = {"dropped_columns": drop_cols}
            set_active_df(new_df, version_name=version_name, diff=diff)
            st.success(f"Applied to Active dataset → {new_df.shape[0]} rows × {new_df.shape[1]} columns")
        else:
            set_all_active_df(drop_cols, base_version_name=version_name)
            st.success("Applied to ALL datasets (columns dropped where present).")

    st.markdown("---")
    export_target = st.radio("Export", options=["Active dataset", "All datasets"], horizontal=True)
    export_fmt = st.selectbox("Export format", ["csv", "parquet"])
    file_name = st.text_input("Export file name (base, without extension)", value="cleaned_dataset")

    if export_target == "Active dataset":
        if st.button("Generate downloadable file (Active)"):
            out = export_df(get_active_df(), fmt=export_fmt)
            st.download_button(
                label=f"Download {export_fmt.upper()} (Active)",
                data=out,
                file_name=f"{file_name}.{export_fmt}",
                mime="text/csv" if export_fmt == "csv" else "application/octet-stream",
                use_container_width=True,
            )
    else:
        st.info("Download each dataset below:")
        for name, item in datasets.items():
            out = export_df(item["active"], fmt=export_fmt)
            st.download_button(
                label=f"⬇️ {name}",
                data=out,
                file_name=f"{file_name}__{name}.{export_fmt}",
                mime="text/csv" if export_fmt == "csv" else "application/octet-stream",
                use_container_width=True,
                key=f"dl_{name}_{export_fmt}"
            )

    st.markdown("---")
    undo_target = st.radio("Undo scope", options=["Active dataset", "All datasets"], horizontal=True)
    if st.button("Undo last version change"):
        msg = undo_last_version(active_only=(undo_target == "Active dataset"))
        if "Reverted" in msg:
            st.warning(msg)
        else:
            st.info(msg)

# -----------------------
# Chart renderers (middle pane)
# -----------------------
def render_boxplots(df: pd.DataFrame, groupby: str | None, dataset_name: str = ""):
    nums = numeric_columns(df)
    if not nums:
        st.info("No numeric columns available for box plots.")
        return
    for col in nums:
        fig, ax = plt.subplots(figsize=(6, 4))
        if groupby and groupby in df.columns:
            sns.boxplot(x=df[groupby], y=df[col], ax=ax)
            ax.set_title(f"{dataset_name} – {col} by {groupby}")
            ax.set_xlabel(groupby)
            ax.set_ylabel(col)
        else:
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"{dataset_name} – {col} distribution")
            ax.set_ylabel(col)
        st.pyplot(fig, clear_figure=True)

def render_corr_matrix(df: pd.DataFrame, method: str = "pearson", mask_upper: bool = True, annot: bool = False, dataset_name: str = ""):
    nums = numeric_columns(df)
    if len(nums) < 2:
        st.info("Need at least 2 numeric columns for correlation.")
        return
    corr = df[nums].corr(method=method)
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=annot, fmt=".2f", ax=ax)
    ax.set_title(f"{dataset_name} – Correlation Matrix ({method})")
    st.pyplot(fig, clear_figure=True)

def render_bar_chart(df: pd.DataFrame, x_cat: str, y_num: str | None, agg: str, hue: str | None, top_n: int, sort_desc: bool, dataset_name: str = ""):
    if x_cat is None or x_cat not in df.columns:
        st.info(f"[{dataset_name}] Select a categorical column for X (not found).")
        return
    temp = df.copy()
    if y_num:
        if y_num not in temp.columns:
            st.info(f"[{dataset_name}] Y column '{y_num}' not found.")
            return
        grp_cols = [x_cat] + ([hue] if hue and hue in temp.columns else [])
        aggfunc = {"sum": "sum", "mean": "mean", "count": "count"}[agg]
        g = temp.groupby(grp_cols, dropna=False)[y_num].agg(aggfunc).reset_index(name="value")
    else:
        grp_cols = [x_cat] + ([hue] if hue and hue in temp.columns else [])
        g = temp.groupby(grp_cols, dropna=False).size().reset_index(name="value")

    # Order & top N
    if sort_desc:
        g = g.sort_values("value", ascending=False)
    if top_n and top_n > 0 and x_cat in g.columns:
        top_keys = g.groupby(x_cat)["value"].sum().sort_values(ascending=False).head(top_n).index
        g = g[g[x_cat].isin(top_keys)]

    fig, ax = plt.subplots(figsize=(7, 5))
    if hue and hue in g.columns:
        sns.barplot(data=g, x=x_cat, y="value", hue=hue, ax=ax)
    else:
        sns.barplot(data=g, x=x_cat, y="value", ax=ax)
    ax.set_xlabel(x_cat)
    ax.set_ylabel(f"{agg}({y_num})" if y_num else "count")
    ax.set_title(f"{dataset_name} – Bar Chart")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, clear_figure=True)

def render_line_chart(df: pd.DataFrame, time_col: str, y_col: str, freq: str, agg: str, groupby: str | None, dataset_name: str = ""):
    if not time_col or not y_col:
        st.info(f"[{dataset_name}] Select a time column and a numeric Y column.")
        return
    if time_col not in df.columns or y_col not in df.columns:
        st.info(f"[{dataset_name}] Columns not found.")
        return

    temp = df.copy()
    temp[time_col] = ensure_datetime(temp[time_col])
    if not np.issubdtype(temp[time_col].dtype, np.datetime64):
        st.warning(f"[{dataset_name}] '{time_col}' could not be parsed as datetime.")
        return

    # Resample / aggregate
    if groupby and groupby in temp.columns:
        grouped = temp.set_index(time_col).groupby(groupby)[y_col]
        pieces = []
        for k, s in grouped:
            try:
                res = getattr(s.resample(freq), agg)()
            except Exception:
                res = s.resample(freq).mean()
            res = res.reset_index().assign(**{groupby: k})
            pieces.append(res)
        agg_df = pd.concat(pieces, ignore_index=True)
    else:
        try:
            agg_df = getattr(temp.set_index(time_col)[y_col].resample(freq), agg)().reset_index()
        except Exception:
            agg_df = temp.set_index(time_col)[y_col].resample(freq).mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    if groupby and groupby in agg_df.columns:
        sns.lineplot(data=agg_df, x=time_col, y=y_col, hue=groupby, ax=ax, marker="o")
    else:
        sns.lineplot(data=agg_df, x=time_col, y=y_col, ax=ax, marker="o")
    ax.set_title(f"{dataset_name} – Line Chart ({agg} per {freq})")
    ax.set_xlabel(time_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig, clear_figure=True)

def render_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None, sample_n: int, alpha: float, dataset_name: str = ""):
    if x not in df.columns or y not in df.columns:
        st.info(f"[{dataset_name}] X/Y columns not found.")
        return
    temp = df[[x, y] + ([hue] if hue and hue in df.columns else [])].dropna().copy()
    if sample_n and len(temp) > sample_n:
        temp = temp.sample(sample_n, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=temp, x=x, y=y, hue=(hue if hue and hue in temp.columns else None), alpha=alpha, ax=ax)
    ax.set_title(f"{dataset_name} – Scatter Plot")
    st.pyplot(fig, clear_figure=True)

def render_bubble(df: pd.DataFrame, x: str, y: str, size_col: str, hue: str | None, sample_n: int, alpha: float, dataset_name: str = ""):
    if x not in df.columns or y not in df.columns or size_col not in df.columns:
        st.info(f"[{dataset_name}] X/Y/Size columns not found.")
        return
    temp = df[[x, y, size_col] + ([hue] if hue and hue in df.columns else [])].dropna().copy()
    # Normalize size for visibility
    s = temp[size_col].astype(float)
    s_min, s_max = (np.percentile(s, 5), np.percentile(s, 95)) if len(s) > 20 else (s.min(), s.max())
    if s_max == s_min:
        sizes = np.full_like(s, 200, dtype=float)
    else:
        sizes = 100 + 900 * (np.clip(s, s_min, s_max) - s_min) / (s_max - s_min)

    temp["_size"] = sizes
    if sample_n and len(temp) > sample_n:
        temp = temp.sample(sample_n, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=temp, x=x, y=y, size="_size", sizes=(50, 1000),
        hue=(hue if hue and hue in temp.columns else None),
        alpha=alpha, ax=ax, legend=True
    )
    ax.set_title(f"{dataset_name} – Bubble Chart")
    st.pyplot(fig, clear_figure=True)

# -----------------------
# Multi‑EDA Summary renderer (per dataset)
# -----------------------
def render_eda_summary(df: pd.DataFrame, dataset_name: str = ""):
    st.markdown(f"### 📄 {dataset_name} – EDA Summary")
    if df.empty:
        st.info("Empty dataset.")
        return
    # Shape & types
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Types:**")
    st.write(df.dtypes.astype(str))

    # Missingness
    st.write("**Missing values (%):**")
    st.write((df.isna().mean() * 100).round(2))

    # Numeric summary
    nums = numeric_columns(df)
    if nums:
        st.write("**Numeric summary (describe):**")
        st.dataframe(df[nums].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns found.")

    # Top categories
    cats = categorical_columns(df)
    if cats:
        st.write("**Top category levels (first 3 columns):**")
        for col in cats[:3]:
            vc = df[col].value_counts(dropna=False).head(10)
            st.write(f"- {col}")
            st.dataframe(vc, use_container_width=True)
    else:
        st.info("No categorical columns found.")

    # Correlation heatmap
    st.write("**Correlation (Pearson):**")
    render_corr_matrix(df, method="pearson", mask_upper=True, annot=False, dataset_name=dataset_name)

    # Preview
    with st.expander("Preview (top 30 rows)"):
        st.dataframe(df.head(30), use_container_width=True)

# -----------------------
# Sidebar: data source & navigation
# -----------------------
st.sidebar.title("📂 Data Sources")
uploads = st.sidebar.file_uploader(
    "Upload multiple CSV / Parquet / Excel files",
    type=["csv", "parquet", "xlsx", "xls"],
    accept_multiple_files=True
)
if uploads:
    ingest_files(uploads)

# Dataset selector
datasets = st.session_state.get(SS["datasets"], {})
if datasets:
    names = sorted(datasets.keys())
    default_idx = names.index(st.session_state.get(SS["active_name"], names[0])) if st.session_state.get(SS["active_name"]) in names else 0
    active_choice = st.sidebar.selectbox("Active dataset", names, index=default_idx)
    st.session_state[SS["active_name"]] = active_choice
    # Shapes overview
    with st.sidebar.expander("Datasets & Shapes"):
        for n in names:
            df = datasets[n]["active"]
            st.write(f"- **{n}**: {df.shape[0]} × {df.shape[1]}")
else:
    st.sidebar.info("Upload one or more files to begin.")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate pages",
    options=[
        "Home",
        "Multi-EDA Summary",       # NEW: runs EDA on all datasets
        "Box Plots",
        "Correlation Matrix",
        "Bar Charts",
        "Line Charts",
        "Scatter Plots",
        "Bubble Charts",
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("Layout:\n- Left: Chart understanding chat\n- Middle: charts & controls\n- Right: data understanding + feature deletion/export")

# -----------------------
# Pages
# -----------------------
if page == "Home":
    left, mid, right = three_pane("🏠 Home")
    df = get_active_df()
    with left:
        chat_panel(CHAT_KEYS["Home"], header="🧠 General EDA Chat", system_prompt="General EDA Q&A about the active dataset.")
    with mid:
        st.subheader("Active Dataset Overview")
        if df.empty:
            st.info("Upload dataset(s) using the sidebar.")
        else:
            st.success(f"Active dataset: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(50), use_container_width=True)
    with right:
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()

elif page == "Multi-EDA Summary":
    left, mid, right = three_pane("🧪 Multi‑EDA Summary (All Uploaded Datasets)")
    with left:
        chat_panel(CHAT_KEYS["Multi-EDA Summary"], header="🧠 EDA Summary Chat", system_prompt="Summarize key insights across all datasets.")
    with mid:
        st.subheader("📊 Batch EDA")
        if not datasets:
            st.info("Upload datasets to run batch EDA.")
        else:
            iterate_datasets(render_eda_summary)
    with right:
        # Show the active dataset's detail + bulk delete/export
        df = get_active_df()
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()

elif page == "Box Plots":
    left, mid, right = three_pane("📦 Box Plots")
    df = get_active_df()
    with left:
        chat_panel(CHAT_KEYS["Box Plots"], header="🧠 Box Plot Chat", system_prompt="Explain outliers (IQR), skewness, and group comparisons.")
    with mid:
        st.subheader("🎛️ Controls")
        run_mode = st.radio("Run on", ["Active dataset", "All datasets"], horizontal=True)
        if not datasets or df.empty:
            st.info("Upload data on Home page.")
        else:
            cats = categorical_columns(df)
            groupby = st.selectbox("Group by (optional)", options=["— None —"] + cats)
            groupby = None if groupby == "— None —" else groupby
            st.divider()
            if run_mode == "Active dataset":
                render_boxplots(df, groupby=groupby, dataset_name=st.session_state[SS["active_name"]])
            else:
                iterate_datasets(render_boxplots, groupby=groupby)
    with right:
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()

elif page == "Correlation Matrix":
    left, mid, right = three_pane("🧩 Correlation Matrix")
    df = get_active_df()
    with left:
        chat_panel(CHAT_KEYS["Correlation Matrix"], header="🧠 Correlation Chat", system_prompt="Discuss correlations and multicollinearity.")
    with mid:
        st.subheader("🎛️ Controls")
        run_mode = st.radio("Run on", ["Active dataset", "All datasets"], horizontal=True)
        if not datasets or df.empty:
            st.info("Upload data on Home page.")
        else:
            method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
            mask_upper = st.checkbox("Mask upper triangle", value=True)
            annot = st.checkbox("Show numeric annotations", value=False)
            st.divider()
            if run_mode == "Active dataset":
                render_corr_matrix(df, method=method, mask_upper=mask_upper, annot=annot, dataset_name=st.session_state[SS["active_name"]])
            else:
                iterate_datasets(render_corr_matrix, method=method, mask_upper=mask_upper, annot=annot)
    with right:
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()

elif page == "Bar Charts":
    left, mid, right = three_pane("📊 Bar Charts (Categories Comparison)")
    df = get_active_df()
    with left:
        chat_panel(CHAT_KEYS["Bar Charts"], header="🧠 Bar Chart Chat", system_prompt="Compare categories and discuss aggregations.")
    with mid:
        st.subheader("🎛️ Controls")
        run_mode = st.radio("Run on", ["Active dataset", "All datasets"], horizontal=True)
        if not datasets or df.empty:
            st.info("Upload data on Home page.")
        else:
            cats = categorical_columns(df)
            nums = numeric_columns(df)
            x_cat = st.selectbox("X (categorical)", options=cats if cats else ["— none —"])
            use_y = st.checkbox("Aggregate a numeric column?", value=True if nums else False)
            y_num = st.selectbox("Y (numeric)", options=nums if nums else ["— none —"]) if use_y and nums else None
            agg = st.selectbox("Aggregation", options=["sum", "mean", "count"], index=0 if use_y else 2)
            hue = st.selectbox("Hue (optional grouping)", options=["— None —"] + cats)
            hue = None if hue == "— None —" else hue
            top_n = st.number_input("Top N categories (by total value)", min_value=0, value=0, step=1, help="0 = show all")
            sort_desc = st.checkbox("Sort descending", value=True)
            st.divider()
            if cats and x_cat != "— none —":
                if run_mode == "Active dataset":
                    render_bar_chart(
                        df, x_cat=x_cat, y_num=(y_num if use_y and y_num in nums else None),
                        agg=agg, hue=hue, top_n=top_n, sort_desc=sort_desc,
                        dataset_name=st.session_state[SS["active_name"]]
                    )
                else:
                    iterate_datasets(
                        render_bar_chart,
                        x_cat=x_cat, y_num=(y_num if use_y else None),
                        agg=agg, hue=hue, top_n=top_n, sort_desc=sort_desc
                    )
            else:
                st.info("No categorical columns found.")
    with right:
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()

elif page == "Line Charts":
    left, mid, right = three_pane("📈 Line Charts (Trends Over Time)")
    df = get_active_df()
    with left:
        chat_panel(CHAT_KEYS["Line Charts"], header="🧠 Line Chart Chat", system_prompt="Explain time trends, seasonality and aggregation.")
    with mid:
        st.subheader("🎛️ Controls")
        run_mode = st.radio("Run on", ["Active dataset", "All datasets"], horizontal=True)
        if not datasets or df.empty:
            st.info("Upload data on Home page.")
        else:
            dt_cols = datetime_columns(df)
            nums = numeric_columns(df)
            time_col = st.selectbox("Time column", options=dt_cols if dt_cols else ["— none —"])
            y_col = st.selectbox("Y (numeric)", options=nums if nums else ["— none —"])
            freq = st.selectbox("Resample frequency", options=["D", "W", "M", "Q", "Y"], index=2,
                                help="D=day, W=week, M=month, Q=quarter, Y=year")
            agg = st.selectbox("Aggregation", options=["sum", "mean", "count"], index=0)
            groupby = st.selectbox("Group by (optional category)", options=["— None —"] + categorical_columns(df))
            groupby = None if groupby == "— None —" else groupby
            st.divider()
            if dt_cols and nums and time_col != "— none —" and y_col != "— none —":
                if run_mode == "Active dataset":
                    render_line_chart(df, time_col=time_col, y_col=y_col, freq=freq, agg=agg, groupby=groupby, dataset_name=st.session_state[SS["active_name"]])
                else:
                    iterate_datasets(render_line_chart, time_col=time_col, y_col=y_col, freq=freq, agg=agg, groupby=groupby)
            else:
                st.info("Select a valid datetime column and a numeric Y.")
    with right:
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()

elif page == "Scatter Plots":
    left, mid, right = three_pane("🔬 Scatter Plots (Relationships)")
    df = get_active_df()
    with left:
        chat_panel(CHAT_KEYS["Scatter Plots"], header="🧠 Scatter Plot Chat", system_prompt="Discuss relationships, clusters, and outliers.")
    with mid:
        st.subheader("🎛️ Controls")
        run_mode = st.radio("Run on", ["Active dataset", "All datasets"], horizontal=True)
        if not datasets or df.empty:
            st.info("Upload data on Home page.")
        else:
            nums = numeric_columns(df)
            cats = categorical_columns(df)
            x = st.selectbox("X (numeric)", options=nums if nums else ["— none —"])
            y = st.selectbox("Y (numeric)", options=nums if nums else ["— none —"])
            hue = st.selectbox("Hue (optional category)", options=["— None —"] + cats)
            hue = None if hue == "— None —" else hue
            sample_n = st.slider("Sample size (for speed)", min_value=500, max_value=100_000, value=5_000, step=500)
            alpha = st.slider("Point opacity", 0.1, 1.0, 0.6, 0.05)
            st.divider()
            if nums and x != "— none —" and y != "— none —":
                if run_mode == "Active dataset":
                    render_scatter(df, x=x, y=y, hue=hue, sample_n=sample_n, alpha=alpha, dataset_name=st.session_state[SS["active_name"]])
                else:
                    iterate_datasets(render_scatter, x=x, y=y, hue=hue, sample_n=sample_n, alpha=alpha)
            else:
                st.info("Select numeric columns for X and Y.")
    with right:
        data_summary_panel(df)
        st.divider()
        feature_delete_panel()

elif page == "Bubble Charts":
    left, mid, right = three_pane("🫧 Bubble Charts (Size‑Encoded Scatter)")
    df = get_active_df()
    with left:
        chat_panel(CHAT_KEYS["Bubble Charts"], header="🧠 Bubble Chart Chat", system_prompt="Discuss size encoding and multi‑dimensional insights.")
    with mid:
        st.subheader("🎛️ Controls")
        run_mode = st.radio("Run on", ["Active dataset", "All datasets"], horizontal=True)
        if not datasets or df.empty:
            st.info("Upload data on Home page.")
        else:
            nums = numeric_columns(df)
            cats = categorical_columns(df)
            x = st.selectbox("X (numeric)", options=nums if nums else ["— none —"])
            y = st.selectbox("Y (numeric)", options=nums if nums else ["— none —"])
            size_col = st.selectbox("Size (numeric)", options=nums if nums else ["— none —"])
            hue = st.selectbox("Hue (optional category)", options=["— None —"] + cats)
            hue = None if hue == "— None —" else hue
            sample_n = st.slider("Sample size (for speed)", min_value=500, max_value=100_000, value=5_000, step=500)
            alpha = st.slider("Point opacity", 0.1, 1.0, 0.6, 0.05)
            st.divider()
            if nums and x != "— none —" and y != "— none —" and size_col != "— none —":
                if run_mode == "Active dataset":
                    render_bubble(df, x=x, y=y, size_col=size_col, hue=hue, sample_n=sample_n, alpha=alpha, dataset_name=st.session_state[SS["active_name"]])
                else:
                    iterate_datasets(render_bubble, x=x, y=y, size_col=size_col, hue=hue, sample_n=sample_n, alpha=alpha)
            else:
                st.info("Select numeric columns for X, Y, and Size.")
    with right:
        data_summary_panel(df)
        st.divider()
