"""
app.py  –  Apple Stock LSTM · Portfolio Demo
============================================
Run with:  streamlit run app.py

Tabs:
  1. Price History        — full AAPL history with train/test split marker
  2. Model Predictions    — per-model actual vs predicted with 95% band
  3. Model Comparison     — all single-step models overlaid + metric table
  4. Hyperparameter Tuning — GridSearchCV heatmap + Keras Tuner results
  5. Methodology          — pipeline, architecture table, caveats, tech stack
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import joblib
import tensorflow as tf
from tensorflow import keras

from data_loader import (
    fetch_aapl_data, load_from_csv, preprocess,
    preprocess_lstm, rescale_close, N_FEATURES, CLOSE_IDX
)
from models import (
    build_lstm, build_rnn_single, build_rnn_multi, build_cnn, build_deep_lstm
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL · Deep Learning Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
h1, h2, h3 { font-family: 'Space Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #0f1729 100%);
    border: 1px solid #2d3a5c;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: #64748b;
    text-transform: uppercase;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-sub { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }

.disclaimer {
    background: #1e1a0e;
    border-left: 3px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #fbbf24;
    margin: 16px 0;
}

.tag {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    margin: 2px;
    font-family: 'Space Mono', monospace;
}

.result-box {
    background: #0f1729;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #7dd3fc;
    margin: 8px 0;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #3b82f6);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Constants & colour palette
# ══════════════════════════════════════════════════════════════════════════════

SAVE_DIR = "saved_models"

COLORS = {
    "actual":       "#38bdf8",
    "lstm_exp":     "#f472b6",
    "lstm_grid":    "#e879f9",
    "rnn_single":   "#fb923c",
    "rnn_multi":    "#fdba74",
    "rnn_grid":     "#fcd34d",
    "cnn":          "#a78bfa",
    "deep":         "#34d399",
}


# ══════════════════════════════════════════════════════════════════════════════
# Plotly theme helper
# ══════════════════════════════════════════════════════════════════════════════

def dark_fig(height: int = 420):
    return go.Figure(layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.9)",
        font=dict(family="Inter", color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1a2540", zeroline=False,
                   showline=True, linecolor="#1e293b"),
        yaxis=dict(gridcolor="#1a2540", zeroline=False,
                   showline=True, linecolor="#1e293b"),
        legend=dict(bgcolor="rgba(15,23,42,0.8)",
                    bordercolor="#334155", borderwidth=1,
                    font=dict(size=11)),
        margin=dict(l=50, r=30, t=50, b=50),
        height=height,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# Cached data & model loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_data():
    try:
        return fetch_aapl_data()
    except Exception:
        st.warning("⚠️  Could not reach Yahoo Finance – using bundled AAPL.csv")
        return load_from_csv("AAPL.csv")


@st.cache_resource(show_spinner=False)
def load_scaler():
    path = os.path.join(SAVE_DIR, "scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _try_load(filename: str):
    """Load a .keras model if the file exists, else return None."""
    path = os.path.join(SAVE_DIR, filename)
    if os.path.exists(path):
        return keras.models.load_model(path, compile=False)
    return None


@st.cache_resource(show_spinner=False)
def load_all_models(_train_data, _a_scaler):
    """
    Load pre-trained models.  If a model file is missing (user hasn't run
    train.py yet) fall back to a fast on-the-fly fit so the app still works.
    """
    models = {}

    # ── Files produced by train.py ────────────────────────────────────────
    models["lstm_exp"]  = _try_load("lstm_exploratory.keras")
    models["lstm_grid"] = _try_load("lstm_best_grid.keras")
    models["rnn_single"]= _try_load("rnn_single_feature.keras")
    models["rnn_multi"] = _try_load("rnn_multi_feature.keras")
    models["rnn_grid"]  = _try_load("rnn_best_grid.keras")
    models["cnn"]       = _try_load("cnn.keras")
    models["deep"]      = _try_load("deep_lstm_2day.keras")

    # ── Fallback: quick train if no saved models found ────────────────────
    cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                        restore_best_weights=True, verbose=0)

    def quick_fit(model, X, y, epochs=20):
        model.fit(X, y, epochs=epochs, batch_size=64,
                  validation_split=0.2, callbacks=[cb], verbose=0)
        return model

    X10, y10 = preprocess_lstm(_train_data, n_inputs=10,
                                n_predictions=1, n_features=N_FEATURES)
    X15, y15 = preprocess_lstm(_train_data, n_inputs=15,
                                n_predictions=1, n_features=N_FEATURES)
    X1f, y1f = preprocess_lstm(pd.DataFrame(_train_data["Close"]),
                                n_inputs=10, n_predictions=1, n_features=1)
    X5f, y5f = preprocess_lstm(_train_data, n_inputs=10,
                                n_predictions=1, n_features=5)
    X7d, y7d = preprocess_lstm(_train_data, n_inputs=7,
                                n_predictions=2, n_features=N_FEATURES)

    if models["lstm_exp"]  is None:
        models["lstm_exp"]  = quick_fit(build_lstm(n_inputs=10, n_features=N_FEATURES,
                                                     neurons=64), X10, y10)
    if models["lstm_grid"] is None:
        models["lstm_grid"] = quick_fit(build_lstm(n_inputs=15, n_features=N_FEATURES,
                                                     neurons=128), X15, y15)
    if models["rnn_single"] is None:
        models["rnn_single"]= quick_fit(build_rnn_single(n_inputs=10, n_features=1,
                                                           neurons=64), X1f, y1f)
    if models["rnn_multi"] is None:
        models["rnn_multi"] = quick_fit(build_rnn_multi(n_inputs=10, n_features=5,
                                                          neurons=64), X5f, y5f)
    if models["rnn_grid"]  is None:
        models["rnn_grid"]  = quick_fit(build_rnn_multi(n_inputs=15, n_features=N_FEATURES,
                                                          neurons=128), X15, y15)
    if models["cnn"]       is None:
        models["cnn"]       = quick_fit(build_cnn(n_inputs=10, n_features=N_FEATURES),
                                         X10, y10, epochs=15)
    if models["deep"]      is None:
        models["deep"]      = quick_fit(build_deep_lstm(n_inputs=7, n_features=N_FEATURES,
                                                          units=192, dense_units=40,
                                                          n_outputs=2), X7d, y7d, epochs=20)
    return models


# ── Load tuning result CSVs (if they exist) ───────────────────────────────────

def load_csv_if_exists(filename: str) -> pd.DataFrame:
    path = os.path.join(SAVE_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Prediction helpers ────────────────────────────────────────────────────────

def get_predictions(model, data, a_scaler, n_inputs, n_predictions, n_feat):
    X, y = preprocess_lstm(data, n_inputs=n_inputs,
                            n_predictions=n_predictions, n_features=n_feat)
    pred   = model.predict(X, verbose=0)
    y_true = rescale_close(y.flatten(), a_scaler)
    y_pred = rescale_close(pred.flatten(), a_scaler)
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape}


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("🔄  Loading data & models — this may take a moment …"):
    df_raw = load_data()
    train_data, test_data, a_scaler, dates = preprocess(df_raw)

    # If scaler was saved by train.py, use that one (ensures consistency)
    saved_scaler = load_scaler()
    if saved_scaler is not None:
        a_scaler = saved_scaler

    models = load_all_models(train_data, a_scaler)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️  Controls")

    MODEL_OPTIONS = {
        "LSTM — Exploratory (10-step, 6 feat)": ("lstm_exp",  10, 1, N_FEATURES),
        "LSTM — Best Grid Search (15-step, 6 feat)": ("lstm_grid", 15, 1, N_FEATURES),
        "RNN — Single Feature (Close only)":    ("rnn_single", 10, 1, 1),
        "RNN — Multi Feature (5 feat)":         ("rnn_multi",  10, 1, 5),
        "RNN — Best Grid Search (15-step, 6 feat)": ("rnn_grid", 15, 1, N_FEATURES),
        "CNN (10-step, 6 feat)":                ("cnn",        10, 1, N_FEATURES),
        "Deep LSTM 2-day (7-step, 6 feat)":     ("deep",        7, 2, N_FEATURES),
    }

    selected_label = st.selectbox("Primary model", list(MODEL_OPTIONS.keys()))
    sel_key, sel_nin, sel_npred, sel_nfeat = MODEL_OPTIONS[selected_label]

    show_confidence = st.toggle("Show 95% prediction band", value=True)

    st.markdown("---")

    # Architecture summary for selected model
    arch_map = {
        "lstm_exp":   "LSTM(64) → Dense(1)\nOptimiser: adam | Loss: logcosh\nInput: (10, 6)",
        "lstm_grid":  "LSTM(128) → Dense(1)\nOptimiser: adam | Loss: logcosh\nInput: (15, 6)\n✅ GridSearchCV best",
        "rnn_single": "SimpleRNN(64) → Dense(1)\nOptimiser: adam | Loss: logcosh\nInput: (10, 1) — Close only",
        "rnn_multi":  "SimpleRNN(64) → Dense(1)\nOptimiser: adam | Loss: logcosh\nInput: (10, 5)",
        "rnn_grid":   "SimpleRNN(128) → Dense(1)\nOptimiser: adam | Loss: logcosh\nInput: (15, 6)\n✅ GridSearchCV best",
        "cnn":        "Conv1D(64,k=3) → MaxPool → Conv1D(32,k=3)\n→ Dense(128) → Dropout(0.15) → Dense(64) → Dense(1)\nOptimiser: adam | Loss: mse\nInput: (10, 6)",
        "deep":       "LSTM(192) → LSTM(192)\n→ Dense(40,linear) → Dense(40,linear) → Dense(2)\nOptimiser: adam | Loss: logcosh\nInput: (7, 6) | Output: 2-day forecast\n✅ Keras Tuner best_hps",
    }
    st.markdown("### 📐  Architecture")
    st.code(arch_map[sel_key], language=None)

    st.markdown("---")
    st.markdown("### 🔮  Live 2-Day Prediction")
    if st.button("Run Deep LSTM →"):
        last_7 = test_data.iloc[-7:].iloc[:, :N_FEATURES].values
        last_7 = np.expand_dims(last_7, axis=0).astype(np.float32)
        pred_2d = models["deep"].predict(last_7, verbose=0)
        p1 = rescale_close(pred_2d[0, 0:1], a_scaler)[0]
        p2 = rescale_close(pred_2d[0, 1:2], a_scaler)[0]
        last_real = df_raw["Close"].iloc[-1]
        st.metric("Day +1", f"${p1:.2f}",
                  f"{(p1 - last_real) / last_real * 100:+.2f}%")
        st.metric("Day +2", f"${p2:.2f}",
                  f"{(p2 - last_real) / last_real * 100:+.2f}%")
        st.markdown(
            "<div class='disclaimer'>⚠️ Educational use only. "
            "Not financial advice.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("TensorFlow · Keras · Streamlit · Plotly · yfinance")


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
# 📈  Apple Stock · Deep Learning Forecast
<p style="color:#64748b;font-size:0.9rem;margin-top:-10px;">
LSTM · SimpleRNN · 1-D CNN · GridSearchCV · Keras Tuner Hyperband &nbsp;|&nbsp; AAPL 1980 – present
</p>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='disclaimer'>⚠️  <b>Educational project only.</b>  "
    "Models predict on historical patterns and have no knowledge of fundamentals, "
    "news, or macro events.  Do <em>not</em> use for investment decisions.</div>",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price History",
    "🔮 Model Predictions",
    "📉 Model Comparison",
    "🔍 Hyperparameter Tuning",
    "🧪 Methodology",
])


# ─── Tab 1 · Price History ────────────────────────────────────────────────────
with tab1:
    st.markdown("### AAPL Closing Price — Full History")

    fig1 = dark_fig(450)
    fig1.add_trace(go.Scatter(
        x=df_raw["Date"], y=df_raw["Close"],
        mode="lines", line=dict(color=COLORS["actual"], width=1.4),
        name="Close Price",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
    ))

    split_date = dates.iloc[round(len(dates) * 0.8)]
    fig1.add_vrect(
        x0=split_date, x1=df_raw["Date"].iloc[-1],
        fillcolor="rgba(248,113,113,0.06)", line_width=0,
        annotation_text="Test set (20%)",
        annotation_position="top left",
        annotation_font_color="#f87171",
        annotation_font_size=11,
    )
    fig1.update_layout(title="AAPL Closing Price (1980 – present)",
                        xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    latest    = df_raw["Close"].iloc[-1]
    ath       = df_raw["Close"].max()
    yr        = df_raw["Date"].iloc[-1].year
    ytd_start = df_raw[df_raw["Date"].dt.year == yr]["Close"].iloc[0]
    ytd_ret   = (latest - ytd_start) / ytd_start * 100

    for col, label, val, sub in [
        (c1, "Latest Close",    f"${latest:.2f}",  ""),
        (c2, "All-Time High",   f"${ath:.2f}",     ""),
        (c3, "YTD Return",
             f"<span style='color:{'#34d399' if ytd_ret>0 else '#f87171'}'>"
             f"{ytd_ret:+.1f}%</span>", ""),
        (c4, "Training Days",   f"{len(train_data):,}", "80 % of history"),
    ]:
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{val}</div>"
                f"<div class='metric-sub'>{sub}</div>"
                f"</div>", unsafe_allow_html=True)


# ─── Tab 2 · Model Predictions ───────────────────────────────────────────────
with tab2:
    st.markdown(f"### {selected_label}")
    st.markdown(f"Test-set predictions vs actual closing price")

    # Use the right test sequences depending on model
    if sel_key == "rnn_single":
        _test_df = pd.DataFrame(test_data["Close"])
    else:
        _test_df = test_data

    y_true, y_pred = get_predictions(
        models[sel_key], _test_df, a_scaler,
        n_inputs=sel_nin, n_predictions=sel_npred, n_features=sel_nfeat)

    # Align dates to test window
    n_plot = min(len(y_true), len(y_pred))
    dates_test = dates.iloc[-len(test_data) + sel_nin:].reset_index(drop=True)
    dates_test = dates_test.iloc[:n_plot]
    y_true = y_true[:n_plot]
    y_pred = y_pred[:n_plot]

    fig2 = dark_fig(480)
    fig2.add_trace(go.Scatter(
        x=dates_test, y=y_true,
        mode="lines", line=dict(color=COLORS["actual"], width=2),
        name="Actual",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=dates_test, y=y_pred,
        mode="lines", line=dict(color=COLORS.get(sel_key, "#f472b6"),
                                 width=1.5, dash="dot"),
        name="Predicted",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
    ))

    if show_confidence:
        resid_std = float(np.std(y_true - y_pred))
        upper = y_pred + 1.96 * resid_std
        lower = y_pred - 1.96 * resid_std
        fig2.add_trace(go.Scatter(
            x=pd.concat([dates_test, dates_test[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(244,114,182,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Band",
            showlegend=True,
        ))

    fig2.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig2, use_container_width=True)

    metrics = compute_metrics(y_true, y_pred)
    mc1, mc2, mc3 = st.columns(3)
    for col, k, fmt in [(mc1, "MSE", "{:.6f}"), (mc2, "RMSE", "${:.4f}"), (mc3, "MAPE", "{:.3f}%")]:
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{k}</div>"
                f"<div class='metric-value'>{fmt.format(metrics[k])}</div>"
                f"</div>", unsafe_allow_html=True)

    # Training loss curves (if history CSV exists)
    history_file_map = {
        "lstm_exp":  "history_lstm_exploratory.csv",
        "lstm_grid": "history_lstm_exploratory.csv",   # best_grid reuses
        "rnn_single":"history_rnn_single.csv",
        "rnn_multi": "history_rnn_multi.csv",
        "rnn_grid":  "history_rnn_multi.csv",
        "cnn":       "history_cnn.csv",
        "deep":      "history_deep_lstm_2day.csv",
    }
    hist_df = load_csv_if_exists(history_file_map.get(sel_key, ""))
    if not hist_df.empty and "loss" in hist_df.columns:
        st.markdown("#### Training & Validation Loss")
        fig_loss = dark_fig(260)
        fig_loss.add_trace(go.Scatter(
            y=hist_df["loss"], mode="lines",
            line=dict(color="#38bdf8", width=1.5), name="Train loss"))
        if "val_loss" in hist_df.columns:
            fig_loss.add_trace(go.Scatter(
                y=hist_df["val_loss"], mode="lines",
                line=dict(color="#f472b6", width=1.5, dash="dash"),
                name="Val loss"))
        fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss",
                                title="Model Accuracy — Loss Curves")
        st.plotly_chart(fig_loss, use_container_width=True)


# ─── Tab 3 · Model Comparison ─────────────────────────────────────────────────
with tab3:
    st.markdown("### All Models — Side-by-Side on Test Set")
    st.caption("All single-step models use 10-step look-back and 6 features "
               "(except RNN single-feature which uses Close only).")

    compare_models = [
        ("LSTM Exploratory",  "lstm_exp",  10, 1, N_FEATURES, test_data),
        ("LSTM Grid Best",    "lstm_grid", 15, 1, N_FEATURES, test_data),
        ("RNN Single Feat",   "rnn_single",10, 1, 1,
         pd.DataFrame(test_data["Close"])),
        ("RNN Multi Feat",    "rnn_multi", 10, 1, 5,          test_data),
        ("RNN Grid Best",     "rnn_grid",  15, 1, N_FEATURES, test_data),
        ("CNN",               "cnn",       10, 1, N_FEATURES, test_data),
    ]

    # Common test ground truth (10-step, 6-feature)
    _, y_te_real_10 = get_predictions(models["lstm_exp"], test_data,
                                       a_scaler, 10, 1, N_FEATURES)
    dates_te10 = dates.iloc[-len(test_data) + 10:].reset_index(drop=True)

    fig3 = dark_fig(520)
    fig3.add_trace(go.Scatter(
        x=dates_te10[:len(y_te_real_10)],
        y=y_te_real_10,
        mode="lines", line=dict(color=COLORS["actual"], width=2.2),
        name="Actual",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
    ))

    metrics_rows = []
    for label, key, n_in, n_pr, n_ft, df_t in compare_models:
        yt, yp = get_predictions(models[key], df_t, a_scaler, n_in, n_pr, n_ft)
        n = min(len(dates_te10), len(yp))
        fig3.add_trace(go.Scatter(
            x=dates_te10[:n], y=yp[:n],
            mode="lines",
            line=dict(color=COLORS.get(key, "#94a3b8"), width=1.2, dash="dot"),
            name=label,
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
        ))
        m = compute_metrics(y_te_real_10[:n], yp[:n])
        metrics_rows.append({"Model": label, **{k: round(v, 6) for k, v in m.items()}})

    fig3.update_layout(xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig3, use_container_width=True)

    # Metrics table
    mdf = pd.DataFrame(metrics_rows).set_index("Model")
    st.dataframe(
        mdf.style
           .highlight_min(axis=0, props="background-color:#14532d;color:white")
           .format({"MSE": "{:.6f}", "RMSE": "${:.4f}", "MAPE": "{:.3f}%"}),
        use_container_width=True,
    )

    # RMSE bar chart
    fig_bar = dark_fig(320)
    bar_colors = [COLORS.get(r[1], "#94a3b8") for r in compare_models]
    fig_bar.add_trace(go.Bar(
        x=mdf.index, y=mdf["RMSE"],
        marker_color=bar_colors,
        text=[f"${v:.4f}" for v in mdf["RMSE"]],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig_bar.update_layout(title="Test-set RMSE by Model (lower = better)",
                           yaxis_title="RMSE (USD)", showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)


# ─── Tab 4 · Hyperparameter Tuning ───────────────────────────────────────────
with tab4:
    st.markdown("### Hyperparameter Tuning Results")
    st.markdown(
        "Results from the two tuning stages in the notebook: "
        "**GridSearchCV** (optimiser × neurons for LSTM and RNN) and "
        "**Keras Tuner Hyperband** (architecture search for 2-day LSTM).")

    # ── GridSearchCV results ──────────────────────────────────────────────
    st.markdown("#### GridSearchCV — Optimiser × Neurons (cv=2, epochs=10)")
    st.caption("Searches `optimiser ∈ ['adam','sgd']` × `neurons ∈ [32,64,128]` "
               "for both LSTM and RNN.  Full parameter set also includes "
               "adagrad, adadelta, rmsprop (commented out in notebook for runtime).")

    gs_lstm = load_csv_if_exists("gridsearch_lstm.csv")
    gs_rnn  = load_csv_if_exists("gridsearch_rnn.csv")

    if not gs_lstm.empty and not gs_rnn.empty:
        col_gs1, col_gs2 = st.columns(2)

        for col, gs_df, title in [
            (col_gs1, gs_lstm, "LSTM GridSearchCV"),
            (col_gs2, gs_rnn,  "RNN  GridSearchCV"),
        ]:
            with col:
                st.markdown(f"**{title}**")
                # Pivot for heatmap: rows=neurons, cols=optimiser
                pivot = gs_df.pivot_table(
                    index="param_neurons",
                    columns="param_optimiser",
                    values="mean_test_score",
                    aggfunc="mean"
                )
                # mean_test_score is negative MSE from sklearn; negate for display
                pivot = -pivot

                fig_hm = dark_fig(280)
                fig_hm.add_trace(go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=[str(v) for v in pivot.index.tolist()],
                    colorscale="Viridis",
                    reversescale=True,
                    text=np.round(pivot.values, 8),
                    texttemplate="%{text:.2e}",
                    showscale=True,
                    colorbar=dict(title="MSE", tickfont=dict(size=9)),
                ))
                fig_hm.update_layout(
                    title=title,
                    xaxis_title="Optimiser",
                    yaxis_title="Neurons",
                    margin=dict(l=50, r=30, t=50, b=50),
                )
                st.plotly_chart(fig_hm, use_container_width=True)

        # Best results box — replicating notebook print statement:
        # "LSTM Best params {'neurons': 128, 'optimiser': 'adam'} with Mean Square Root Error ..."
        best_lstm_row = gs_lstm.loc[gs_lstm["mean_test_score"].idxmax()]
        best_rnn_row  = gs_rnn.loc[gs_rnn["mean_test_score"].idxmax()]

        st.markdown(
            f"<div class='result-box'>"
            f"LSTM Best params  {{'neurons': {int(best_lstm_row['param_neurons'])}, "
            f"'optimiser': '{best_lstm_row['param_optimiser']}'}}  "
            f"with Mean Square Root Error {best_lstm_row['mean_test_score']:.6e}<br><br>"
            f"RNN  Best model   {{'neurons': {int(best_rnn_row['param_neurons'])}, "
            f"'optimiser': '{best_rnn_row['param_optimiser']}'}}  "
            f"with Mean Square Root Error {best_rnn_row['mean_test_score']:.6e}"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("GridSearchCV results not found. Run `python train.py` to generate them.")
        st.markdown("**Expected output from notebook:**")
        st.code(
            "LSTM Best params {'neurons': 128, 'optimiser': 'adam'} "
            "with Mean Square Root Error -3.169875917308218e-07\n"
            "RNN  Best model  {'neurons': 128, 'optimiser': 'adam'} "
            "with Mean Square Root Error -3.5331660219739547e-06",
            language=None
        )

    st.markdown("---")

    # ── Keras Tuner — single-step results ────────────────────────────────
    st.markdown("#### Keras Tuner Hyperband — Single-Step LSTM")
    st.caption("Searches `units ∈ [32, 448]` (step 32) across input sizes "
               "[10, 15, 20, 24, 25, 30].  Full sweep over sizes 10–30 "
               "takes ~200 min (notebook comment); we use the reduced set.")

    tuner_single = load_csv_if_exists("tuner_single_step_results.csv")
    if not tuner_single.empty:
        fig_t1 = dark_fig(320)
        fig_t1.add_trace(go.Scatter(
            x=tuner_single["inputs"].astype(str),
            y=tuner_single["val_loss"],
            mode="markers+lines",
            marker=dict(size=10, color="#38bdf8",
                        symbol="circle", line=dict(color="#0ea5e9", width=1)),
            line=dict(color="#38bdf8", width=1.5),
            text=[f"units={int(u)}" for u in tuner_single["units"]],
            textposition="top center",
            textfont=dict(size=10, color="#94a3b8"),
            name="Best val_loss",
            hovertemplate="n_inputs=%{x}<br>val_loss=%{y:.4e}<br>%{text}<extra></extra>",
        ))
        fig_t1.update_layout(
            title="Tuner Best val_loss by Input Size",
            xaxis_title="Look-back window (n_inputs)",
            yaxis_title="Best val_loss",
        )
        st.plotly_chart(fig_t1, use_container_width=True)
        st.dataframe(tuner_single.sort_values("val_loss").reset_index(drop=True),
                     use_container_width=True)
    else:
        st.info("Tuner results not found.  Run `python train.py` to generate them.")

    st.markdown("---")

    # ── Keras Tuner — 2-day prediction ───────────────────────────────────
    st.markdown("#### Keras Tuner Hyperband — 2-Day LSTM")
    st.caption("`units ∈ [192, 252]` (step 32), `dense ∈ [10, 50]` (step 10), "
               "stacked two-layer LSTM, max_epochs=20, objective=val_mse.")

    best_hps_path = os.path.join(SAVE_DIR, "best_hps_2day.json")
    if os.path.exists(best_hps_path):
        with open(best_hps_path) as f:
            best_hps = json.load(f)
        st.markdown(
            f"<div class='result-box'>"
            f"The hyperparameter search is complete.<br>"
            f"Optimal units in LSTM layers: <b>{best_hps['units']}</b><br>"
            f"Optimal units in dense layers: <b>{best_hps['dense']}</b>"
            f"</div>", unsafe_allow_html=True)

        history_2d = load_csv_if_exists("history_deep_lstm_2day.csv")
        if not history_2d.empty:
            fig_2d = dark_fig(280)
            fig_2d.add_trace(go.Scatter(
                y=history_2d["loss"], mode="lines",
                line=dict(color="#34d399", width=1.5), name="Train loss"))
            if "val_loss" in history_2d.columns:
                fig_2d.add_trace(go.Scatter(
                    y=history_2d["val_loss"], mode="lines",
                    line=dict(color="#f59e0b", width=1.5, dash="dash"),
                    name="Val loss"))
            fig_2d.update_layout(title="Deep LSTM 2-Day — Training Curves",
                                  xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig_2d, use_container_width=True)
    else:
        st.info("2-day tuner results not found.  Run `python train.py` to generate them.")


# ─── Tab 5 · Methodology ──────────────────────────────────────────────────────
with tab5:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("""
### Pipeline

**1. Data Collection**
- Source: Yahoo Finance via `yfinance` (auto_adjust=False)
- Features: Open, High, Low, **Close**, Adj Close, Volume — **6 features** matching
  the original notebook's AAPL.csv
- Range: 1980-12-12 → present (~11 K trading days)

**2. Preprocessing**
```python
a_scaler = MinMaxScaler(feature_range=(0, 1))
apple_norm = pd.DataFrame(a_scaler.fit_transform(apple_norm),
                           columns=apple_norm.columns)
train_data = apple_norm[:round(len(apple_norm["Open"]) * 0.8)]
test_data  = apple_norm[round(len(apple_norm["Open"]) * 0.8):]
```

**3. Sequence Construction (`preprocess_lstm`)**
```python
for i in range(n_inputs, len(df) - n_predictions + 1):
    X_train.append(df.iloc[i-n_inputs:i, 0:n_features])
    y_train.append(df["Close"][i : i + n_predictions])
```

**4. Inverse Scaling**
```python
y_rescaled = (y_scaled - a_scaler.min_[3]) / a_scaler.scale_[3]
```
index 3 = Close in [Open, High, Low, **Close**, Adj Close, Volume]

**5. Models**

| Model | Input shape | Output | Notes |
|-------|------------|--------|-------|
| LSTM exploratory | (10, 6) | 1 | units=64, adam, logcosh |
| LSTM grid best | (15, 6) | 1 | units=128, adam — GridSearchCV winner |
| RNN single feat | (10, 1) | 1 | Close only, units=64 |
| RNN multi feat | (10, 5) | 1 | n_features=5 |
| RNN grid best | (15, 6) | 1 | units=128, adam — GridSearchCV winner |
| CNN | (10, 6) | 1 | Conv1D(64,k=3)→Pool→Conv1D(32,k=3)→Dense |
| Deep LSTM 2-day | (7, 6) | 2 | 2×LSTM(192)→2×Dense(40) — Tuner winner |

**6. Tuning stages**
- GridSearchCV: `optimiser=['adam','sgd'] × neurons=[32,64,128]`, cv=2
- Keras Tuner Hyperband: units 32–448, input sizes 10–30 (single-step)
- Keras Tuner Hyperband: units 192–252, dense 10–50 (2-day model)
        """)

    with col_b:
        st.markdown("""
### Why LSTM Outperforms SimpleRNN

LSTM cells have three gating mechanisms — input gate, forget gate, output gate —
that let the network selectively retain or discard information across long
sequences.  SimpleRNNs suffer from **vanishing gradients** when sequences grow
long, causing them to lose patterns that occurred many steps ago.

Stock prices exhibit long-range dependencies (multi-month trends, quarterly
earnings cycles) that LSTM is specifically designed to capture.

### Limitations & Caveats

- **No look-ahead bias** — strict chronological 80/20 split; scaler is fit
  on training data only
- Models capture *statistical patterns only* — no awareness of earnings
  announcements, macro events, or market sentiment
- MAPE can be misleading near zero; RMSE in USD is more interpretable
- Real-world deployment would require continuous retraining as the
  data distribution shifts over time

### Reproducing Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full training + tuning (one-time, ~30–60 min on CPU)
python train.py

# 3. Launch the portfolio app
streamlit run app.py
```
        """)

        st.markdown("### Tech Stack")
        tags = [
            "Python 3.11", "TensorFlow 2.x", "Keras", "keras-tuner",
            "scikit-learn", "Streamlit", "Plotly", "yfinance",
            "NumPy", "pandas", "joblib",
        ]
        st.markdown(
            " ".join(f"<span class='tag'>{t}</span>" for t in tags),
            unsafe_allow_html=True)
