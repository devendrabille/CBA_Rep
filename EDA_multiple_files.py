# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io, re
from datetime import datetime

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Agentic EDA (All Files, Same UI)", layout="wide")

# -----------------------
# Session state keys
# -----------------------
SS = {
    "datasets": "datasets",   # dict: name -> {"original": df, "active": df, "versions": []}
}

# -----------------------
# Utilities
# -----------------------
def keyify(*parts):
    """Create a safe Streamlit key from parts (dataset + chart page + element)."""
    s = "__".join(str(p) for p in parts)
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)

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
    for f in files:
        try:
            df = read_file(f)
            name = _ensure_unique_name(datasets.keys(), f.name)
            datasets[name] = {"original": df.copy(), "active": df.copy(), "versions": []}
        except Exception as e:
            st.sidebar.error(f"Failed to read {f.name}: {e}")

def get_df(dataset_name) -> pd.DataFrame:
    datasets = st.session_state.get(SS["datasets"], {})
    if dataset_name in datasets:
        return datasets[dataset_name]["active"]
    return pd.DataFrame()

def set_df(dataset_name, df, version_name=None, diff=None):
    datasets = st.session_state.get(SS["datasets"], {})
    if dataset_name in datasets:
        datasets[dataset_name]["active"] = df
        if version_name:
            datasets[dataset_name]["versions"].append(
                {"name": version_name, "df": df.copy(), "diff": diff}
            )

def undo_last_version(dataset_name):
    datasets = st.session_state.get(SS["datasets"], {})
    if dataset_name not in datasets:
        return "Dataset not found."
    versions = datasets[dataset_name]["versions"]
    if not versions:
        datasets[dataset_name]["active"] = datasets[dataset_name]["original"].copy()
        return "No versions to undo. Reverted to original."
    versions.pop()
    if versions:
        datasets[dataset_name]["active"] = versions[-1]["df"].copy()
        return "Reverted to previous version."
    else:
        datasets[dataset_name]["active"] = datasets[dataset_name]["original"].copy()
        return "Reverted to original."

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
# Column helpers
# -----------------------
def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

def datetime_columns(df: pd.DataFrame):
    dt_cols = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            dt_cols.append(col)
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            pd.to_datetime(df[col], errors="raise")
            dt_cols.append(col)
        except Exception:
            pass
    return list(dict.fromkeys(dt_cols))

def ensure_datetime(series: pd.Series):
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    try:
        return pd.to_datetime(series)
    except Exception:
        return series

# -----------------------
# Layout helper
# -----------------------
def three_pane(title: str):
    st.title(title)
    left, mid, right = st.columns([1.1, 1.8, 1.1])
    return left, mid, right

# -----------------------
# Chat panel (stubbed; wire Azure OpenAI later)
# -----------------------
def llm_call(system_prompt: str, messages: list) -> str:
    return "🔎 (Stub) I would explain this chart based on the controls and dataset context."

def chat_panel(state_key: str, header: str, system_prompt: str = None,
               input_key: str = None, clear_key: str = None):
    """Chat panel with UNIQUE keys for chat_input and clear button."""
    st.subheader(header)

    # Persist messages in session_state under a dataset+chart specific key
    msgs = st.session_state.setdefault(state_key, [])

    # Render past messages
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # ✅ Unique key for chat_input
    user_input = st.chat_input("Ask about the chart…", key=input_key)
    if user_input:
        msgs.append({"role": "user", "content": user_input})
        response = llm_call(system_prompt or "", msgs)   # <-- wire Azure OpenAI here
        msgs.append({"role": "assistant", "content": response})
        st.session_state[state_key] = msgs

    # ✅ Unique key for "Clear chat" button
    cols_btn = st.columns([1, 1])
    with cols_btn[0]:
        if st.button("Clear chat", key=clear_key):
            st.session_state[state_key] = []
            st.experimental_rerun()
    with cols_btn[1]:
        st.caption("Chat is scoped to this dataset & chart.")

def chat_for(dataset_name: str, chart_slug: str, header: str, prompt: str):
    """Convenience wrapper to create unique keys per dataset x chart."""
    return chat_panel(
        state_key=keyify(dataset_name, "chat", chart_slug),
        header=header,
        system_prompt=prompt,
        input_key=keyify(dataset_name, "chat_input", chart_slug),
        clear_key=keyify(dataset_name, "chat_clear", chart_slug),
    )

# -----------------------
# Right pane: data summary + feature deletion/export
# -----------------------
def data_summary_panel(df: pd.DataFrame):
    st.subheader("📚 Data Understanding")
    if df.empty:
        st.info("Empty dataset.")
        return
    st.write(f"**Rows:** {df.shape[0]} **Columns:** {df.shape[1]}")
    st.markdown("**Types:**")
    st.write(df.dtypes.astype(str))
    st.markdown("**Missing values (%):**")
    st.write((df.isna().mean() * 100).round(2))
    with st.expander("Preview (top 30 rows)"):
        st.dataframe(df.head(30), use_container_width=True)

def feature_delete_panel(dataset_name: str):
    st.subheader("🧹 Delete Features & Export")
    df = get_df(dataset_name)
    if df.empty:
        st.info("No data to manage.")
        return

    drop_cols = st.multiselect("Select features to delete", options=list(df.columns), default=[],
                               key=keyify(dataset_name, "del_multiselect"))
    if drop_cols:
        st.caption(f"Dropping {len(drop_cols)} column(s): {drop_cols}")

    version_name = st.text_input(
        "Version name",
        value=f"{dataset_name}__drop__{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        key=keyify(dataset_name, "del_version_name")
    )
    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("Apply deletion", key=keyify(dataset_name, "apply_delete")):
            new_df = df.drop(columns=drop_cols) if drop_cols else df.copy()
            diff = {"dropped_columns": drop_cols}
            set_df(dataset_name, new_df, version_name=version_name, diff=diff)
            st.success(f"Applied to **{dataset_name}** → {new_df.shape[0]} × {new_df.shape[1]}")

    with cols[1]:
        if st.button("Undo last change", key=keyify(dataset_name, "undo")):
            msg = undo_last_version(dataset_name)
            st.warning(msg)

    with cols[2]:
        if st.button("Reset to original", key=keyify(dataset_name, "reset")):
            datasets = st.session_state.get(SS["datasets"], {})
            if dataset_name in datasets:
                datasets[dataset_name]["active"] = datasets[dataset_name]["original"].copy()
                datasets[dataset_name]["versions"].clear()
                st.warning("Reset to original.")

    st.markdown("---")
    export_fmt = st.selectbox("Export format", ["csv", "parquet"], key=keyify(dataset_name, "export_fmt"))
    file_name = st.text_input("Export file name (without extension)", value=f"{dataset_name}__cleaned",
                              key=keyify(dataset_name, "export_name"))
    out = export_df(get_df(dataset_name), fmt=export_fmt)
    st.download_button(
        label=f"⬇️ Download {export_fmt.upper()}",
        data=out,
        file_name=f"{file_name}.{export_fmt}",
        mime="text/csv" if export_fmt == "csv" else "application/octet-stream",
        use_container_width=True,
        key=keyify(dataset_name, "download_btn", export_fmt)
    )

# -----------------------
# Chart renderers
# -----------------------
def render_boxplots(df: pd.DataFrame, groupby: str | None):
    nums = numeric_columns(df)
    if not nums:
        st.info("No numeric columns for box plots.")
        return
    for col in nums:
        fig, ax = plt.subplots(figsize=(6, 4))
        if groupby and groupby in df.columns:
            sns.boxplot(x=df[groupby], y=df[col], ax=ax)
            ax.set_title(f"{col} by {groupby}")
            ax.set_xlabel(groupby); ax.set_ylabel(col)
        else:
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"{col} distribution")
            ax.set_ylabel(col)
        st.pyplot(fig, clear_figure=True)

def render_corr_matrix(df: pd.DataFrame, method: str = "pearson", mask_upper: bool = True, annot: bool = False):
    nums = numeric_columns(df)
    if len(nums) < 2:
        st.info("Need ≥ 2 numeric columns for correlation.")
        return
    corr = df[nums].corr(method=method)
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=annot, fmt=".2f", ax=ax)
    ax.set_title(f"Correlation ({method})")
    st.pyplot(fig, clear_figure=True)

def render_bar_chart(df: pd.DataFrame, x_cat: str, y_num: str | None, agg: str, hue: str | None, top_n: int, sort_desc: bool):
    if x_cat not in df.columns:
        st.info("Pick a valid categorical column for X.")
        return
    temp = df.copy()
    if y_num and y_num in temp.columns:
        grp_cols = [x_cat] + ([hue] if hue and hue in temp.columns else [])
        aggfunc = {"sum": "sum", "mean": "mean", "count": "count"}[agg]
        g = temp.groupby(grp_cols, dropna=False)[y_num].agg(aggfunc).reset_index(name="value")
    else:
        grp_cols = [x_cat] + ([hue] if hue and hue in temp.columns else [])
        g = temp.groupby(grp_cols, dropna=False).size().reset_index(name="value")

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
    ax.set_title("Bar Chart")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, clear_figure=True)

def render_line_chart(df: pd.DataFrame, time_col: str, y_col: str, freq: str, agg: str, groupby: str | None):
    if time_col not in df.columns or y_col not in df.columns:
        st.info("Select a valid time column and numeric Y.")
        return
    temp = df.copy()
    temp[time_col] = ensure_datetime(temp[time_col])
    if not np.issubdtype(temp[time_col].dtype, np.datetime64):
        st.warning(f"Column '{time_col}' could not be parsed as datetime.")
        return
    if groupby and groupby in temp.columns:
        grouped = temp.set_index(time_col).groupby(groupby)[y_col]
        pieces = []
        for k, s in grouped:
            try:
                res = getattr(s.resample(freq), agg)()
            except Exception:
                res = s.resample(freq).mean()
            pieces.append(res.reset_index().assign(**{groupby: k}))
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
    ax.set_title(f"Line Chart ({agg} per {freq})")
    ax.set_xlabel(time_col); ax.set_ylabel(y_col)
    st.pyplot(fig, clear_figure=True)

def render_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None, sample_n: int, alpha: float):
    if x not in df.columns or y not in df.columns:
        st.info("Select numeric columns for X and Y.")
        return
    temp = df[[x, y] + ([hue] if hue and hue in df.columns else [])].dropna().copy()
    if sample_n and len(temp) > sample_n:
        temp = temp.sample(sample_n, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=temp, x=x, y=y, hue=(hue if hue and hue in temp.columns else None), alpha=alpha, ax=ax)
    ax.set_title("Scatter Plot")
    st.pyplot(fig, clear_figure=True)

def render_bubble(df: pd.DataFrame, x: str, y: str, size_col: str, hue: str | None, sample_n: int, alpha: float):
    if x not in df.columns or y not in df.columns or size_col not in df.columns:
        st.info("Select numeric columns for X, Y, and Size.")
        return
    temp = df[[x, y, size_col] + ([hue] if hue and hue in df.columns else [])].dropna().copy()
    s = temp[size_col].astype(float)
    s_min, s_max = (np.percentile(s, 5), np.percentile(s, 95)) if len(s) > 20 else (s.min(), s.max())
    sizes = np.full_like(s, 200, dtype=float) if s_max == s_min else 100 + 900 * (np.clip(s, s_min, s_max) - s_min) / (s_max - s_min)
    temp["_size"] = sizes
    if sample_n and len(temp) > sample_n:
        temp = temp.sample(sample_n, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=temp, x=x, y=y, size="_size", sizes=(50, 1000),
        hue=(hue if hue and hue in temp.columns else None),
        alpha=alpha, ax=ax, legend=True
    )
    ax.set_title("Bubble Chart")
    st.pyplot(fig, clear_figure=True)

# -----------------------
# Sidebar: uploads + tips
# -----------------------
st.sidebar.title("📂 Upload Datasets")
uploads = st.sidebar.file_uploader(
    "Upload multiple CSV / Parquet / Excel files",
    type=["csv", "parquet", "xlsx", "xls"],
    accept_multiple_files=True
)
if uploads:
    ingest_files(uploads)

datasets = st.session_state.get(SS["datasets"], {})
if not datasets:
    st.sidebar.info("Upload files to begin. Parquet requires `pyarrow`; Excel requires `openpyxl`.")

with st.sidebar.expander("Help / Troubleshooting"):
    st.write("- If charts don't update after deletion, change a control or rerun (↻).")
    st.write("- For large files, use sampling on Scatter/Bubble.")
    st.write("- Clear cache via **Settings → Clear cache** if data seems stale.")

# -----------------------
# Main: tabs per dataset
# -----------------------
if not datasets:
    st.info("No datasets loaded yet. Upload multiple files from the sidebar.")
else:
    dataset_names = sorted(datasets.keys())
    tabs = st.tabs(dataset_names)

    for tab, dname in zip(tabs, dataset_names):
        with tab:
            df = get_df(dname)
            st.markdown(f"### 📁 Dataset: **{dname}**  |  Shape: `{df.shape[0]} × {df.shape[1]}`")

            # Sub-tabs per chart page
            chart_tabs = st.tabs([
                "Overview",
                "Box Plots",
                "Correlation Matrix",
                "Bar Charts",
                "Line Charts",
                "Scatter Plots",
                "Bubble Charts",
            ])

            # -------- Overview (3-pane) --------
            with chart_tabs[0]:
                left, mid, right = three_pane(f"Overview – {dname}")
                with left:
                    chat_for(dname, "overview", "🧠 Overview Chat",
                             "Summarize dataset shape, types, missingness, and top-level observations.")
                with mid:
                    st.subheader("Dataset Preview & Quick Stats")
                    if df.empty:
                        st.info("Empty dataset.")
                    else:
                        st.dataframe(df.head(50), use_container_width=True)
                        nums = numeric_columns(df)
                        if nums:
                            st.markdown("**Numeric summary (describe):**")
                            st.dataframe(df[nums].describe().T, use_container_width=True)
                        cats = categorical_columns(df)
                        if cats:
                            st.markdown("**Top category levels (first 3 categorical cols):**")
                            for col in cats[:3]:
                                vc = df[col].value_counts(dropna=False).head(10)
                                st.write(f"- {col}")
                                st.dataframe(vc, use_container_width=True)
                with right:
                    data_summary_panel(df)
                    st.divider()
                    feature_delete_panel(dname)

            # -------- Box Plots (3-pane) --------
            with chart_tabs[1]:
                left, mid, right = three_pane(f"Box Plots – {dname}")
                with left:
                    chat_for(dname, "box", "🧠 Box Plot Chat",
                             "Explain distribution, outliers (IQR), skewness, and group comparisons.")
                with mid:
                    st.subheader("🎛️ Controls")
                    if df.empty:
                        st.info("Upload data.")
                    else:
                        cats = categorical_columns(df)
                        groupby = st.selectbox("Group by (optional)", options=["— None —"] + cats,
                                               key=keyify(dname, "box_group"))
                        groupby = None if groupby == "— None —" else groupby
                        st.divider()
                        render_boxplots(df, groupby=groupby)
                with right:
                    data_summary_panel(df)
                    st.divider()
                    feature_delete_panel(dname)

            # -------- Correlation Matrix (3-pane) --------
            with chart_tabs[2]:
                left, mid, right = three_pane(f"Correlation – {dname}")
                with left:
                    chat_for(dname, "corr", "🧠 Correlation Chat",
                             "Discuss correlation strength, multicollinearity and feature selection implications.")
                with mid:
                    st.subheader("🎛️ Controls")
                    if df.empty:
                        st.info("Upload data.")
                    else:
                        method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True,
                                          key=keyify(dname, "corr_method"))
                        mask_upper = st.checkbox("Mask upper triangle", value=True,
                                                 key=keyify(dname, "corr_mask"))
                        annot = st.checkbox("Show numeric annotations", value=False,
                                            key=keyify(dname, "corr_annot"))
                        st.divider()
                        render_corr_matrix(df, method=method, mask_upper=mask_upper, annot=annot)
                with right:
                    data_summary_panel(df)
                    st.divider()
                    feature_delete_panel(dname)

            # -------- Bar Charts (3-pane) --------
            with chart_tabs[3]:
                left, mid, right = three_pane(f"Bar Charts – {dname}")
                with left:
                    chat_for(dname, "bar", "🧠 Bar Chart Chat",
                             "Compare categories with sum/mean/count; discuss distributions and top-N.")
                with mid:
                    st.subheader("🎛️ Controls")
                    if df.empty:
                        st.info("Upload data.")
                    else:
                        cats = categorical_columns(df)
                        nums = numeric_columns(df)
                        x_cat = st.selectbox("X (categorical)", options=cats if cats else ["— none —"],
                                             key=keyify(dname, "bar_x"))
                        use_y = st.checkbox("Aggregate a numeric column?", value=bool(nums),
                                            key=keyify(dname, "bar_usey"))
                        y_num = st.selectbox("Y (numeric)", options=nums if nums else ["— none —"],
                                             key=keyify(dname, "bar_y")) if use_y and nums else None
                        agg = st.selectbox("Aggregation", options=["sum", "mean", "count"],
                                           index=0 if use_y else 2, key=keyify(dname, "bar_agg"))
                        hue = st.selectbox("Hue (optional grouping)", options=["— None —"] + cats,
                                           key=keyify(dname, "bar_hue"))
                        hue = None if hue == "— None —" else hue
                        top_n = st.number_input("Top N categories", min_value=0, value=0, step=1,
                                                key=keyify(dname, "bar_top"))
                        sort_desc = st.checkbox("Sort descending", value=True, key=keyify(dname, "bar_sort"))
                        st.divider()
                        if cats and x_cat != "— none —":
                            render_bar_chart(df, x_cat=x_cat, y_num=(y_num if use_y else None),
                                             agg=agg, hue=hue, top_n=top_n, sort_desc=sort_desc)
                        else:
                            st.info("No categorical columns found.")
                with right:
                    data_summary_panel(df)
                    st.divider()
                    feature_delete_panel(dname)

            # -------- Line Charts (3-pane) --------
            with chart_tabs[4]:
                left, mid, right = three_pane(f"Line Charts – {dname}")
                with left:
                    chat_for(dname, "line", "🧠 Line Chart Chat",
                             "Explain time trends, seasonality, resampling and aggregation.")
                with mid:
                    st.subheader("🎛️ Controls")
                    if df.empty:
                        st.info("Upload data.")
                    else:
                        dt_cols = datetime_columns(df)
                        nums = numeric_columns(df)
                        time_col = st.selectbox("Time column", options=dt_cols if dt_cols else ["— none —"],
                                                key=keyify(dname, "line_time"))
                        y_col = st.selectbox("Y (numeric)", options=nums if nums else ["— none —"],
                                             key=keyify(dname, "line_y"))
                        freq = st.selectbox("Resample frequency", options=["D", "W", "M", "Q", "Y"], index=2,
                                            key=keyify(dname, "line_freq"))
                        agg = st.selectbox("Aggregation", options=["sum", "mean", "count"], index=0,
                                           key=keyify(dname, "line_agg"))
                        groupby = st.selectbox("Group by (optional category)", options=["— None —"] + categorical_columns(df),
                                               key=keyify(dname, "line_group"))
                        groupby = None if groupby == "— None —" else groupby
                        st.divider()
                        if dt_cols and nums and time_col != "— none —" and y_col != "— none —":
                            render_line_chart(df, time_col=time_col, y_col=y_col, freq=freq, agg=agg, groupby=groupby)
                        else:
                            st.info("Select a valid datetime column and a numeric Y.")
                with right:
                    data_summary_panel(df)
                    st.divider()
                    feature_delete_panel(dname)

            # -------- Scatter Plots (3-pane) --------
            with chart_tabs[5]:
                left, mid, right = three_pane(f"Scatter Plots – {dname}")
                with left:
                    chat_for(dname, "scatter", "🧠 Scatter Plot Chat",
                             "Discuss relationships, clusters, outliers and segmentation.")
                with mid:
                    st.subheader("🎛️ Controls")
                    if df.empty:
                        st.info("Upload data.")
                    else:
                        nums = numeric_columns(df)
                        cats = categorical_columns(df)
                        x = st.selectbox("X (numeric)", options=nums if nums else ["— none —"],
                                         key=keyify(dname, "scatter_x"))
                        y = st.selectbox("Y (numeric)", options=nums if nums else ["— none —"],
                                         key=keyify(dname, "scatter_y"))
                        hue = st.selectbox("Hue (optional category)", options=["— None —"] + cats,
                                           key=keyify(dname, "scatter_hue"))
                        hue = None if hue == "— None —" else hue
                        sample_n = st.slider("Sample size", min_value=500, max_value=100_000, value=5_000, step=500,
                                             key=keyify(dname, "scatter_sample"))
                        alpha = st.slider("Point opacity", 0.1, 1.0, 0.6, 0.05, key=keyify(dname, "scatter_alpha"))
                        st.divider()
                        if nums and x != "— none —" and y != "— none —":
                            render_scatter(df, x=x, y=y, hue=hue, sample_n=sample_n, alpha=alpha)
                        else:
                            st.info("Select numeric columns for X and Y.")
                with right:
                    data_summary_panel(df)
                    st.divider()
                    feature_delete_panel(dname)

            # -------- Bubble Charts (3-pane) --------
            with chart_tabs[6]:
                left, mid, right = three_pane(f"Bubble Charts – {dname}")
                with left:
                    chat_for(dname, "bubble", "🧠 Bubble Chart Chat",
                             "Discuss size encoding, multi-dimensional insights and scaling.")
                with mid:
                    st.subheader("🎛️ Controls")
                    if df.empty:
                        st.info("Upload data.")
                    else:
                        nums = numeric_columns(df)
                        cats = categorical_columns(df)
                        x = st.selectbox("X (numeric)", options=nums if nums else ["— none —"],
                                         key=keyify(dname, "bubble_x"))
                        y = st.selectbox("Y (numeric)", options=nums if nums else ["— none —"],
                                         key=keyify(dname, "bubble_y"))
                        size_col = st.selectbox("Size (numeric)", options=nums if nums else ["— none —"],
                                                key=keyify(dname, "bubble_size"))
                        hue = st.selectbox("Hue (optional category)", options=["— None —"] + cats,
                                           key=keyify(dname, "bubble_hue"))
                        hue = None if hue == "— None —" else hue
                        sample_n = st.slider("Sample size", min_value=500, max_value=100_000, value=5_000, step=500,
                                             key=keyify(dname, "bubble_sample"))
                        alpha = st.slider("Point opacity", 0.1, 1.0, 0.6, 0.05, key=keyify(dname, "bubble_alpha"))
                        st.divider()
                        if nums and x != "— none —" and y != "— none —" and size_col != "— none —":
                            render_bubble(df, x=x, y=y, size_col=size_col, hue=hue, sample_n=sample_n, alpha=alpha)
                        else:
                            st.info("Select numeric columns for X, Y, and Size.")
                with right:
                    data_summary_panel(df)
                    st.divider()
