# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Page setup & minimal styling
# ---------------------------
st.set_page_config(page_title="Agentic EDA (Plotly + Forecasts)", layout="wide", page_icon="🧠")

CSS = """
.block-container { padding-top: 1rem; }
.stMarkdown, .stDataFrame, .stPlotlyChart {
  background: #0f1628; border-radius: 12px; padding: 12px;
}
.stButton>button { border-radius: 8px; }
[data-testid="column"]>div:has(.stSubheader) { margin-top: -8px; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ---------------------------
# Session state initialization
# ---------------------------
DEFAULT_KEYS = {
    "df": None,
    "selected_features": [],
    "deleted_features": set(),
    "current_chart": None,
    "insights": [],
    "chat_history": []
}
for k, v in DEFAULT_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def add_insight(text, source="system"):
    st.session_state.insights.append({"text": text, "source": source})

def record_chat(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

# ---------------------------
# Layout helpers
# ---------------------------
def three_column_shell():
    left, mid, right = st.columns([0.9, 1.4, 0.9], gap="small")
    return left, mid, right

def left_chat_panel():
    st.subheader("💬 Chat & Insights")
    query = st.chat_input("Ask about the data or chart...")
    if query:
        record_chat("user", query)
        df = st.session_state.df
        if df is not None:
            add_insight(f"Dataset: {len(df):,} rows × {df.shape[1]} columns.", "agent")
        record_chat("assistant", "Captured your query. Insights updated.")
    if st.session_state.insights:
        with st.expander("Insights feed", expanded=False):
            for ins in st.session_state.insights[-10:]:
                st.caption(f"{ins['source']}: {ins['text']}")
    return query

def right_data_panel(df: pd.DataFrame):
    st.subheader("📚 Data Understanding")
    st.caption(f"Rows: {len(df):,} • Columns: {df.shape[1]}")
    with st.expander("Schema", expanded=False):
        st.write(df.dtypes.to_frame("dtype"))
    with st.expander("Missingness", expanded=False):
        st.write(df.isna().sum())
    with st.expander("Sample (head 20)", expanded=False):
        st.write(df.head(20))

# ---------------------------
# Data utilities
# ---------------------------
@st.cache_data(show_spinner=True)
def load_csv(upload):
    return pd.read_csv(upload)

@st.cache_data(show_spinner=True)
def generate_sample_timeseries(n_days=900, seed=42):
    """Generate a realistic multi-feature time series dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    trend = np.linspace(100, 300, n_days)
    weekly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = rng.normal(0, 10, n_days)

    regions = ["North", "South", "East", "West"]
    region_series = rng.choice(regions, size=n_days, replace=True)
    campaigns = ["None", "A", "B"]
    campaign_series = rng.choice(campaigns, size=n_days, replace=True, p=[0.6, 0.2, 0.2])

    base = trend + weekly + noise
    mkt_spend = (rng.gamma(shape=5, scale=8, size=n_days) *
                 (1 + (campaign_series != "None") * rng.uniform(0.3, 0.8, n_days)))
    users = np.maximum(0, base + mkt_spend * rng.uniform(0.5, 1.5, n_days))
    price_per_user = rng.normal(1.5, 0.1, n_days)
    revenue = np.maximum(0, users * price_per_user + rng.normal(0, 20, n_days))

    df = pd.DataFrame({
        "date": dates,
        "region": region_series,
        "campaign": campaign_series,
        "users": users.round(2),
        "revenue": revenue.round(2),
        "marketing_spend": mkt_spend.round(2)
    })
    return df

def numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def categorical_columns(df):
    return [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])]

def delete_feature(name: str):
    if name not in st.session_state.deleted_features:
        st.session_state.deleted_features.add(name)

def apply_deletions_to_df():
    df = st.session_state.df
    if df is not None and st.session_state.deleted_features:
        remaining = [c for c in df.columns if c not in st.session_state.deleted_features]
        st.session_state.df = df[remaining]
        st.session_state.selected_features = [
            f for f in st.session_state.selected_features if f in remaining
        ]
        st.success("Deletions applied")

def export_csv(df: pd.DataFrame, filename="cleaned.csv"):
    st.download_button("⬇️ Export CSV", data=df.to_csv(index=False), file_name=filename, mime="text/csv")

# ---------------------------
# Plotly chart builders
# ---------------------------
def plot_histogram(df: pd.DataFrame, feature: str):
    fig = px.histogram(df, x=feature, nbins=40, opacity=0.85)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark")
    fig.update_traces(marker_color="#60a5fa")
    return fig

def plot_scatter(df: pd.DataFrame, x: str, y: str, color: str | None):
    fig = px.scatter(df, x=x, y=y, color=color if color else None, opacity=0.75)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark")
    return fig

def plot_timeseries(df_ts: pd.DataFrame, metric: str, show_ma=True, anomalies_flag="is_anomaly"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ts["date"], y=df_ts[metric], mode="lines", name="Actual",
        line=dict(color="#60a5fa")
    ))
    if show_ma and "moving_avg" in df_ts.columns:
        fig.add_trace(go.Scatter(
            x=df_ts["date"], y=df_ts["moving_avg"], mode="lines", name="Moving Avg",
            line=dict(color="#f59e0b", width=2, dash="solid")
        ))
    if anomalies_flag in df_ts.columns and df_ts[anomalies_flag].any():
        anom = df_ts[df_ts[anomalies_flag]]
        fig.add_trace(go.Scatter(
            x=anom["date"], y=anom[metric], mode="markers", name="Anomaly",
            marker=dict(color="#ef4444", size=9)
        ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ---------------------------
# Forecasting utilities (no external libs)
# ---------------------------
def forecast_naive(train: pd.Series, horizon: int) -> np.ndarray:
    last_val = float(train.iloc[-1])
    return np.full(horizon, last_val)

def forecast_seasonal_naive(train: pd.Series, horizon: int, seasonal_period: int) -> np.ndarray:
    if len(train) < seasonal_period:
        # Fallback to naive if not enough data
        return forecast_naive(train, horizon)
    last_season = train.iloc[-seasonal_period:].to_numpy()
    # Repeat last season values to cover horizon
    reps = int(np.ceil(horizon / seasonal_period))
    return np.tile(last_season, reps)[:horizon].astype(float)

def forecast_moving_average(train: pd.Series, horizon: int, window: int) -> np.ndarray:
    window = max(1, min(window, len(train)))
    avg = float(train.iloc[-window:].mean())
    return np.full(horizon, avg)

def forecast_ses(train: pd.Series, horizon: int, alpha: float) -> np.ndarray:
    """Simple Exponential Smoothing (no trend/season)."""
    alpha = float(np.clip(alpha, 0.01, 0.99))
    s = float(train.iloc[0])
    for y in train:
        s = alpha * float(y) + (1 - alpha) * s
    return np.full(horizon, s)

def make_forecast(model_name: str, train: pd.Series, horizon: int,
                  seasonal_period: int | None = None, ma_window: int | None = None,
                  alpha: float | None = None) -> np.ndarray:
    if model_name == "Naive (Last)":
        return forecast_naive(train, horizon)
    elif model_name == "Seasonal Naive":
        period = seasonal_period or 7
        return forecast_seasonal_naive(train, horizon, int(period))
    elif model_name == "Moving Average":
        w = ma_window or 7
        return forecast_moving_average(train, horizon, int(w))
    elif model_name == "Simple Exponential Smoothing":
        a = alpha or 0.3
        return forecast_ses(train, horizon, float(a))
    else:
        # Default fallback
        return forecast_naive(train, horizon)

# Helper: default seasonal period by frequency
def default_seasonal_period(freq: str) -> int:
    # D -> 7, W -> 52, M -> 12, Q -> 4
    return {"D": 7, "W": 52, "M": 12, "Q": 4}.get(freq, 7)

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Distribution", "Relationships", "Time Series", "Model Performance"],
    index=0
)

# ---------------------------
# Page: Overview
# ---------------------------
if page == "Overview":
    left, mid, right = three_column_shell()

    with left:
        st.subheader("Dataset")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            st.session_state.df = load_csv(uploaded)
            st.toast("Dataset loaded ✅")
        if st.button("📈 Load sample time series"):
            st.session_state.df = generate_sample_timeseries()
            st.success("Sample time series dataset loaded")

    with mid:
        st.subheader("Summary")
        df = st.session_state.df
        if df is not None:
            st.write(df.describe(include="all").T)

    with right:
        df = st.session_state.df
        if df is not None:
            right_data_panel(df)
            export_csv(df)

# ---------------------------
# Page: Distribution
# ---------------------------
elif page == "Distribution":
    df = st.session_state.df
    if df is None:
        st.warning("No dataset loaded. Upload a CSV or load the sample in Overview.")
        st.stop()

    left, mid, right = three_column_shell()

    with left:
        _ = left_chat_panel()
        num_cols = numeric_columns(df)
        if not num_cols:
            st.error("No numeric columns available for distribution charts.")
            st.stop()

        feature = st.selectbox("Feature", options=num_cols)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Mark feature for deletion"):
                delete_feature(feature)
                st.toast(f"Marked '{feature}' for deletion")
        with col2:
            if st.button("✅ Apply deletions"):
                apply_deletions_to_df()

    with mid:
        st.subheader("Histogram (Plotly)")
        fig = plot_histogram(df, feature)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        right_data_panel(df)

# ---------------------------
# Page: Relationships
# ---------------------------
elif page == "Relationships":
    df = st.session_state.df
    if df is None:
        st.warning("No dataset loaded. Upload a CSV or load the sample in Overview.")
        st.stop()

    left, mid, right = three_column_shell()

    with left:
        _ = left_chat_panel()
        num_cols = numeric_columns(df)
        cat_cols = categorical_columns(df)
        if len(num_cols) < 2:
            st.error("Need at least two numeric columns for a scatter plot.")
            st.stop()

        x = st.selectbox("X", num_cols)
        y = st.selectbox("Y", [c for c in num_cols if c != x] or num_cols)
        color_opt = st.selectbox("Color (optional)", ["(none)"] + cat_cols)
        color = None if color_opt == "(none)" else color_opt

    with mid:
        st.subheader("Scatter Plot (Plotly)")
        fig = plot_scatter(df, x, y, color)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        right_data_panel(df)

# ---------------------------
# Page: Time Series
# ---------------------------
elif page == "Time Series":
    df = st.session_state.df
    if df is None:
        st.warning("No dataset loaded. Upload a CSV or load the sample in Overview.")
        st.info("TIP: The sample dataset includes 'date', 'region', 'campaign', 'users', 'revenue', 'marketing_spend'.")
        st.stop()

    if "date" not in df.columns:
        st.error("Time series requires a 'date' column. Please upload a dataset with a date field or use the sample.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    left, mid, right = three_column_shell()

    with left:
        _ = left_chat_panel()
        regions = sorted(df["region"].dropna().unique()) if "region" in df.columns else []
        campaigns = sorted(df["campaign"].dropna().unique()) if "campaign" in df.columns else []

        region = st.selectbox("Region (filter)", ["(all)"] + regions) if regions else "(all)"
        campaign = st.selectbox("Campaign (filter)", ["(all)"] + campaigns) if campaigns else "(all)"

        filt = pd.Series(True, index=df.index)
        if regions and region != "(all)":
            filt &= (df["region"] == region)
        if campaigns and campaign != "(all)":
            filt &= (df["campaign"] == campaign)
        df_f = df.loc[filt].copy()

        candidates = [c for c in numeric_columns(df_f) if c != "date"]
        default_metric_idx = candidates.index("revenue") if "revenue" in candidates else 0
        metric = st.selectbox("Metric", candidates, index=default_metric_idx)

        freq_label = st.selectbox("Aggregate frequency", ["D (daily)", "W (weekly)", "M (monthly)", "Q (quarterly)"], index=1)
        freq_map = {"D (daily)": "D", "W (weekly)": "W", "M (monthly)": "M", "Q (quarterly)": "Q"}
        freq = freq_map[freq_label]

        win = st.slider("Moving average window (periods)", 3, 60, 14)
        z_thresh = st.slider("Anomaly z-score threshold", 1.5, 4.0, 2.5)

        # Forecast model selection (preview)
        model_name = st.selectbox("Forecast model (preview)", ["Naive (Last)", "Seasonal Naive", "Moving Average", "Simple Exponential Smoothing"], index=1)
        sp_default = default_seasonal_period(freq)
        seasonal_period = st.number_input("Seasonal period (for Seasonal Naive)", min_value=2, max_value=365, value=sp_default, step=1)
        ma_window = st.slider("MA window (for Moving Average)", 2, 60, 7)
        alpha = st.slider("SES alpha (0.01–0.99)", 0.01, 0.99, 0.3)

    with mid:
        st.subheader(f"Time Series • {metric}")
        df_ts = (df_f[["date", metric]]
                 .set_index("date")
                 .resample(freq)
                 .sum()
                 .reset_index()
                 .sort_values("date"))

        # Moving average
        if len(df_ts) >= win:
            df_ts["moving_avg"] = df_ts[metric].rolling(window=win, min_periods=max(1, win//2)).mean()
        else:
            df_ts["moving_avg"] = np.nan

        # Rolling z-score anomalies
        roll_mean = df_ts[metric].rolling(window=win, min_periods=max(1, win//2)).mean()
        roll_std = df_ts[metric].rolling(window=win, min_periods=max(1, win//2)).std()
        z = (df_ts[metric] - roll_mean) / (roll_std.replace(0, np.nan))
        df_ts["is_anomaly"] = z.abs() > z_thresh

        # Plot
        fig = plot_timeseries(df_ts, metric, show_ma=True, anomalies_flag="is_anomaly")
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Points", value=f"{len(df_ts):,}")
        with colB:
            st.metric("Anomalies", value=f"{int(df_ts['is_anomaly'].sum()):,}")
        with colC:
            st.metric("Avg Level", value=f"{df_ts[metric].mean():.2f}")

    with right:
        right_data_panel(df_f)
        st.caption("ℹ️ Anomalies use rolling z-scores vs moving average.")

        # Quick forecast preview (next period)
        with st.expander("Quick forecast preview (chosen model)", expanded=False):
            train = df_ts[metric]
            horizon_preview = 1
            fc = make_forecast(
                model_name,
                train=train,
                horizon=horizon_preview,
                seasonal_period=seasonal_period,
                ma_window=ma_window,
                alpha=alpha
            )
            next_val = float(fc[-1]) if len(fc) else np.nan
            st.write(f"Forecast for next {freq} period: **{next_val:.2f}**")

        export_csv(df_ts.rename(columns={metric: f"{metric}_{freq}"}), filename=f"{metric}_{freq}.csv")

# ---------------------------
# Page: Model Performance (with model selection)
# ---------------------------
elif page == "Model Performance":
    df = st.session_state.df
    if df is None or "date" not in df.columns:
        st.warning("Requires a dataset with a 'date' column. Load the sample (Overview) to proceed.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    left, mid, right = three_column_shell()

    with left:
        _ = left_chat_panel()
        candidates = [c for c in numeric_columns(df) if c != "date"]
        metric = st.selectbox("Metric", candidates, index=(candidates.index("revenue") if "revenue" in candidates else 0))
        freq = st.selectbox("Frequency for evaluation", ["D", "W", "M", "Q"], index=1)
        horizon = st.slider("Test horizon (last N periods)", 8, 60, 30)

        # Forecast model selection
        model_name = st.selectbox("Forecast model", ["Naive (Last)", "Seasonal Naive", "Moving Average", "Simple Exponential Smoothing"], index=1)
        sp_default = default_seasonal_period(freq)
        seasonal_period = st.number_input("Seasonal period (Seasonal Naive)", min_value=2, max_value=365, value=sp_default, step=1)
        ma_window = st.slider("MA window (Moving Average)", 2, 60, 7)
        alpha = st.slider("SES alpha (0.01–0.99)", 0.01, 0.99, 0.3)

    with mid:
        st.subheader(f"Forecast Evaluation • {model_name} • {metric}/{freq}")
        df_ts = (df[["date", metric]]
                 .set_index("date")
                 .resample(freq)
                 .sum()
                 .reset_index()
                 .sort_values("date"))

        if len(df_ts) < horizon + 5:
            st.error("Not enough periods for the selected horizon.")
            st.stop()

        # Train/Test split
        train = df_ts.iloc[:-horizon].copy()
        test = df_ts.iloc[-horizon:].copy()

        # Forecast
        fc = make_forecast(
            model_name,
            train=train[metric],
            horizon=horizon,
            seasonal_period=seasonal_period,
            ma_window=ma_window,
            alpha=alpha
        )
        test["forecast"] = fc

        # Metrics
        mae = float(np.mean(np.abs(test[metric] - test["forecast"])))
        mape = float(np.mean(np.abs((test[metric] - test["forecast"]) / np.maximum(1e-8, test[metric]))) * 100)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE", value=f"{mae:.2f}")
        with col2:
            st.metric("MAPE", value=f"{mape:.2f}%")

        # Plot: train (faint), test actual vs forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train["date"], y=train[metric],
                                 mode="lines", name="Train (actual)",
                                 line=dict(color="#94a3b8", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=test["date"], y=test[metric],
                                 mode="lines", name="Test (actual)",
                                 line=dict(color="#2563eb", width=3)))
        fig.add_trace(go.Scatter(x=test["date"], y=test["forecast"],
                                 mode="lines", name="Forecast",
                                 line=dict(color="#f59e0b", width=3)))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          template="plotly_dark",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        right_data_panel(df)
        st.caption("Baseline models are simple and fast. Replace with advanced models as needed.")
        export_csv(test[["date", metric, "forecast"]], filename=f"{metric}_{freq}_forecast_eval.csv")

# ---------------------------
# Footer note
# ---------------------------
st.caption("🧠 Agentic EDA • Plotly + Forecasts • Single-file demo")