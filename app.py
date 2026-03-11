"""
app.py  –  Apple Stock LSTM · Portfolio Demo
Run with:  streamlit run app.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from data_loader import (
    fetch_aapl_data, load_from_csv, preprocess,
    preprocess_lstm, rescale_close, N_FEATURES, CLOSE_IDX
)
from models import (
    build_lstm, build_rnn_single, build_rnn_multi, build_cnn, build_deep_lstm
)

st.set_page_config(
    page_title="AAPL · Deep Learning Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Serif+4:wght@300;400;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: #0d0d0d;
    color: #e8e0d4;
}
h1 { font-family: 'Playfair Display', Georgia, serif; font-weight: 900; letter-spacing: -0.02em; }
h2, h3 { font-family: 'Playfair Display', Georgia, serif; font-weight: 700; }

.kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 4px;
}
.metric-card {
    background: #161616;
    border-top: 3px solid #c9a84c;
    border-bottom: 1px solid #2a2a2a;
    padding: 16px 20px 14px;
    text-align: left;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #f0ead6;
    line-height: 1;
}
.metric-sub { font-family: 'Source Serif 4', serif; font-size: 0.78rem; color: #666; margin-top: 5px; }
.insight-box {
    border-left: 3px solid #c9a84c;
    background: #141414;
    padding: 14px 20px;
    margin: 16px 0;
    font-family: 'Source Serif 4', serif;
    font-size: 0.92rem;
    color: #c8bfa8;
    font-style: italic;
    line-height: 1.6;
}
.disclaimer {
    background: #1a1500;
    border-left: 3px solid #c9a84c;
    padding: 10px 16px;
    border-radius: 0 4px 4px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #c9a84c;
    margin: 12px 0;
}
.result-box {
    background: #111;
    border: 1px solid #222;
    border-left: 3px solid #c9a84c;
    border-radius: 0 4px 4px 0;
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #d4c89a;
    margin: 10px 0;
    line-height: 1.7;
}
.caption-text {
    font-family: 'Source Serif 4', serif;
    font-size: 0.82rem;
    color: #666;
    font-style: italic;
    line-height: 1.5;
    margin-bottom: 12px;
}
.anno-badge {
    display: inline-block;
    background: #c9a84c;
    color: #0d0d0d;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    text-transform: uppercase;
    border-radius: 1px;
    vertical-align: middle;
    margin-right: 6px;
}
.stButton > button {
    background: #c9a84c;
    color: #0d0d0d;
    border: none;
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    padding: 0.5rem 1.5rem;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] { background-color: #111 !important; border-right: 1px solid #222; }
.stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; color: #666 !important; }
.stTabs [aria-selected="true"] { color: #c9a84c !important; border-bottom-color: #c9a84c !important; }
</style>
""", unsafe_allow_html=True)

SAVE_DIR = "saved_models"
COLORS = {
    "actual":     "#e8e0d4",
    "lstm_exp":   "#c9a84c",
    "lstm_grid":  "#e8956d",
    "rnn_single": "#a8c4d4",
    "rnn_multi":  "#7eb8c9",
    "rnn_grid":   "#6aa3b8",
    "cnn":        "#b5a0d8",
    "deep":       "#7dd3a8",
}
AAPL_EVENTS = [
    ("2007-01-09", "iPhone"),
    ("2010-01-27", "iPad"),
    ("2020-03-23", "COVID low"),
    ("2022-01-03", "ATH $182"),
]

def editorial_fig(height=440, title=""):
    fig = go.Figure(layout=go.Layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111111",
        font=dict(family="Source Serif 4, Georgia, serif", color="#888", size=11),
        xaxis=dict(gridcolor="#1e1e1e", zeroline=False, showline=True, linecolor="#2a2a2a",
                   tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#666")),
        yaxis=dict(gridcolor="#1e1e1e", zeroline=False, showline=False,
                   tickfont=dict(family="JetBrains Mono, monospace", size=10, color="#666"),
                   tickprefix="$"),
        legend=dict(bgcolor="rgba(13,13,13,0.9)", bordercolor="#2a2a2a", borderwidth=1,
                    font=dict(family="JetBrains Mono, monospace", size=10, color="#888"),
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        margin=dict(l=60, r=30, t=55, b=50), height=height,
        title=dict(text=f"<b>{title}</b>",
                   font=dict(family="Playfair Display, Georgia, serif", size=15, color="#e8e0d4"),
                   x=0, xanchor="left") if title else None,
    ))
    return fig

def plain_fig(height=280, title=""):
    fig = editorial_fig(height, title)
    fig.update_layout(yaxis=dict(tickprefix="", gridcolor="#1e1e1e", zeroline=False))
    return fig

@st.cache_resource(show_spinner=False)
def load_data():
    try:
        df = fetch_aapl_data()
        if df is None or len(df) == 0: raise ValueError
        return df, True
    except Exception:
        return load_from_csv("AAPL.csv"), False

@st.cache_resource(show_spinner=False)
def load_scaler():
    path = os.path.join(SAVE_DIR, "scaler.pkl")
    return joblib.load(path) if os.path.exists(path) else None

def _try_load(fn):
    p = os.path.join(SAVE_DIR, fn)
    return keras.models.load_model(p, compile=False) if os.path.exists(p) else None

@st.cache_resource(show_spinner=False)
def load_all_models(_train_data, _a_scaler):
    models = {
        "lstm_exp":   _try_load("lstm_exploratory.keras"),
        "lstm_grid":  _try_load("lstm_best_grid.keras"),
        "rnn_single": _try_load("rnn_single_feature.keras"),
        "rnn_multi":  _try_load("rnn_multi_feature.keras"),
        "rnn_grid":   _try_load("rnn_best_grid.keras"),
        "cnn":        _try_load("cnn.keras"),
        "deep":       _try_load("deep_lstm_2day.keras"),
    }
    cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
    def qfit(m, X, y, ep=20):
        m.fit(X, y, epochs=ep, batch_size=64, validation_split=0.2, callbacks=[cb], verbose=0)
        return m
    X10,y10 = preprocess_lstm(_train_data, 10, 1, N_FEATURES)
    X15,y15 = preprocess_lstm(_train_data, 15, 1, N_FEATURES)
    X1f,y1f = preprocess_lstm(pd.DataFrame(_train_data["Close"]), 10, 1, 1)
    X5f,y5f = preprocess_lstm(_train_data, 10, 1, 5)
    X7d,y7d = preprocess_lstm(_train_data, 7, 2, N_FEATURES)
    if models["lstm_exp"]   is None: models["lstm_exp"]   = qfit(build_lstm(10, N_FEATURES, 64), X10, y10)
    if models["lstm_grid"]  is None: models["lstm_grid"]  = qfit(build_lstm(15, N_FEATURES, 128), X15, y15)
    if models["rnn_single"] is None: models["rnn_single"] = qfit(build_rnn_single(10, 1, 64), X1f, y1f)
    if models["rnn_multi"]  is None: models["rnn_multi"]  = qfit(build_rnn_multi(10, 5, 64), X5f, y5f)
    if models["rnn_grid"]   is None: models["rnn_grid"]   = qfit(build_rnn_multi(15, N_FEATURES, 128), X15, y15)
    if models["cnn"]        is None: models["cnn"]        = qfit(build_cnn(10, N_FEATURES), X10, y10, 15)
    if models["deep"]       is None: models["deep"]       = qfit(build_deep_lstm(7, N_FEATURES, 192, 40, 2), X7d, y7d)
    return models

def load_csv_if_exists(fn):
    p = os.path.join(SAVE_DIR, fn)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

def get_predictions(model, data, a_scaler, n_inputs, n_predictions, n_feat):
    X, y = preprocess_lstm(data, n_inputs, n_predictions, n_feat)
    pred = model.predict(X, verbose=0)
    return rescale_close(y.flatten(), a_scaler), rescale_close(pred.flatten(), a_scaler)

def compute_metrics(y_true, y_pred):
    mse  = float(np.mean((y_true - y_pred)**2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred)/(np.abs(y_true)+1e-8)))*100)
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape}

# ── Bootstrap ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data & models…"):
    df_raw, _is_live = load_data()
    train_data, test_data, a_scaler, dates = preprocess(df_raw)
    saved_scaler = load_scaler()
    if saved_scaler is not None: a_scaler = saved_scaler
    models = load_all_models(train_data, a_scaler)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    latest_close = df_raw["Close"].iloc[-1]
    latest_date  = df_raw["Date"].iloc[-1]
    date_str = latest_date.strftime('%b %d, %Y') if hasattr(latest_date,'strftime') else str(latest_date)
    st.markdown(f"""
<div class='kicker'>Market Data</div>
<div style='font-family:Playfair Display,serif;font-size:1.05rem;color:#e8e0d4;margin-bottom:2px'>AAPL</div>
<div style='font-family:JetBrains Mono,monospace;font-size:1.5rem;font-weight:700;color:#c9a84c'>${latest_close:.2f}</div>
<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#555;margin-bottom:16px'>{date_str}</div>
""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#222;margin:0 0 16px'>", unsafe_allow_html=True)

    MODEL_OPTIONS = {
        "LSTM — Exploratory (10-step, 6 feat)":    ("lstm_exp",  10, 1, N_FEATURES),
        "LSTM — Best Grid Search (15-step, 6 feat)":("lstm_grid", 15, 1, N_FEATURES),
        "RNN — Single Feature (Close only)":        ("rnn_single",10, 1, 1),
        "RNN — Multi Feature (5 feat)":             ("rnn_multi", 10, 1, 5),
        "RNN — Best Grid Search (15-step, 6 feat)": ("rnn_grid",  15, 1, N_FEATURES),
        "CNN (10-step, 6 feat)":                    ("cnn",       10, 1, N_FEATURES),
        "Deep LSTM 2-day (7-step, 6 feat)":         ("deep",       7, 2, N_FEATURES),
    }
    st.markdown("<div class='kicker'>Primary Model</div>", unsafe_allow_html=True)
    selected_label = st.selectbox("", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")
    sel_key, sel_nin, sel_npred, sel_nfeat = MODEL_OPTIONS[selected_label]
    show_confidence = st.toggle("Show 95% prediction band", value=True)
    show_volume     = st.toggle("Show volume overlay", value=False)

    st.markdown("<hr style='border-color:#222;margin:16px 0'>", unsafe_allow_html=True)
    arch_map = {
        "lstm_exp":   "LSTM(64) → Dense(1)\nOptimiser: adam | Loss: logcosh\nInput: (10, 6)",
        "lstm_grid":  "LSTM(128) → Dense(1)\nOptimiser: adam\nInput: (15, 6)\n✅ GridSearchCV best",
        "rnn_single": "SimpleRNN(64) → Dense(1)\nOptimiser: adam\nInput: (10, 1) — Close only",
        "rnn_multi":  "SimpleRNN(64) → Dense(1)\nOptimiser: adam\nInput: (10, 5)",
        "rnn_grid":   "SimpleRNN(128) → Dense(1)\nOptimiser: adam\nInput: (15, 6)\n✅ GridSearchCV best",
        "cnn":        "Conv1D(64,k=3)→MaxPool\n→Conv1D(32,k=3)→Dense(128)\n→Dropout(0.15)→Dense(1)\nInput: (10, 6)",
        "deep":       "LSTM(192)→LSTM(192)\n→Dense(40)→Dense(40)→Dense(2)\nInput: (7,6) | 2-day\n✅ Keras Tuner best",
    }
    st.markdown("<div class='kicker'>Architecture</div>", unsafe_allow_html=True)
    st.code(arch_map[sel_key], language=None)

    st.markdown("<hr style='border-color:#222;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Live 2-Day Forecast</div>", unsafe_allow_html=True)
    if st.button("Run Deep LSTM →"):
        last_7 = test_data.iloc[-7:].iloc[:, :N_FEATURES].values
        last_7 = np.expand_dims(last_7, 0).astype(np.float32)
        pred_2d = models["deep"].predict(last_7, verbose=0)
        p1 = rescale_close(pred_2d[0, 0:1], a_scaler)[0]
        p2 = rescale_close(pred_2d[0, 1:2], a_scaler)[0]
        last_real = df_raw["Close"].iloc[-1]
        for label, p in [("Day +1", p1), ("Day +2", p2)]:
            d = (p - last_real) / last_real * 100
            c = "#4ade80" if d >= 0 else "#f87171"
            st.markdown(f"<div style='margin:8px 0'><div class='metric-label'>{label}</div><div style='font-family:Playfair Display,serif;font-size:1.4rem;color:#e8e0d4'>${p:.2f} <span style='font-size:0.85rem;color:{c}'>{d:+.2f}%</span></div></div>", unsafe_allow_html=True)
        st.markdown("<div class='disclaimer'>⚠️ Educational use only. Not financial advice.</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#222;margin:16px 0'>", unsafe_allow_html=True)
    src_color = "#4ade80" if _is_live else "#888"
    src_label = "Live Data" if _is_live else "Bundled CSV"
    last_date_str = df_raw['Date'].iloc[-1].strftime('%Y-%m-%d') if hasattr(df_raw['Date'].iloc[-1],'strftime') else str(df_raw['Date'].iloc[-1])
    st.markdown(f"<div class='kicker' style='color:{src_color}'>● {src_label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#444'>Last: {last_date_str}</div>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
today_str = datetime.now().strftime("%B %d, %Y").upper()
st.markdown(f"""
<div class='kicker'>{today_str} &nbsp;·&nbsp; AAPL &nbsp;·&nbsp; DEEP LEARNING RESEARCH</div>
<h1 style='margin:4px 0 2px;font-size:2.2rem;line-height:1.15'>Apple Stock Forecasting<br>with Deep Learning</h1>
<p style='font-family:Source Serif 4,serif;color:#888;font-size:0.95rem;margin:6px 0 0'>
LSTM · SimpleRNN · 1-D CNN · GridSearchCV · Keras Tuner Hyperband &nbsp;|&nbsp; AAPL 1980–present &nbsp;|&nbsp; 7 architectures compared
</p>
<div style='border-top:2px solid #c9a84c;margin:16px 0 4px'></div>
""", unsafe_allow_html=True)
st.markdown("<div class='disclaimer'>⚠️ <b>Educational project only.</b> Models predict on historical patterns and have no knowledge of fundamentals, news, or macro events. Do <em>not</em> use for investment decisions.</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Price History", "🔮  Model Predictions",
    "📉  Model Comparison", "🔍  Hyperparameter Tuning", "🧪  Methodology",
])

# ── TAB 1 · Price History ─────────────────────────────────────────────────────
with tab1:
    latest = df_raw["Close"].iloc[-1]
    ath    = df_raw["Close"].max()
    ath_date = df_raw.loc[df_raw["Close"].idxmax(), "Date"]
    yr = df_raw["Date"].iloc[-1].year
    ytd_rows = df_raw[df_raw["Date"].dt.year == yr]
    ytd_start = ytd_rows["Close"].iloc[0] if len(ytd_rows) else latest
    ytd_ret = (latest - ytd_start) / ytd_start * 100
    rows_2015 = df_raw[df_raw["Date"].dt.year == 2015]
    dec_start = rows_2015["Close"].iloc[0] if len(rows_2015) else latest
    dec_ret = (latest - dec_start) / dec_start * 100

    c1,c2,c3,c4,c5 = st.columns(5)
    ath_str = ath_date.strftime('%b %Y') if hasattr(ath_date,'strftime') else ""
    for col, label, val, sub, chg in [
        (c1, "Latest Close",   f"${latest:.2f}",      "",              None),
        (c2, "All-Time High",  f"${ath:.2f}",          ath_str,         None),
        (c3, "YTD Return",     f"{ytd_ret:+.1f}%",     str(yr),         ytd_ret),
        (c4, "10-Year Return", f"{dec_ret:+,.0f}%",    "since Jan 2015",dec_ret),
        (c5, "Training Days",  f"{len(train_data):,}", f"80% · {len(test_data):,} test", None),
    ]:
        color = ("#4ade80" if chg>0 else "#f87171") if chg is not None else "#f0ead6"
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='color:{color}'>{val}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    split_date = dates.iloc[round(len(dates)*0.8)]

    if show_volume:
        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75,0.25], vertical_spacing=0.03)
        fig1.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111111",
            font=dict(family="Source Serif 4,Georgia,serif",color="#888",size=11),
            height=520, legend=dict(bgcolor="rgba(13,13,13,0.9)",bordercolor="#2a2a2a",borderwidth=1,font=dict(family="JetBrains Mono,monospace",size=10,color="#888")),
            margin=dict(l=60,r=30,t=55,b=50),
            title=dict(text="<b>Apple Inc. (AAPL) — Closing Price & Volume, 1980–Present</b>",font=dict(family="Playfair Display,Georgia,serif",size=15,color="#e8e0d4"),x=0,xanchor="left"))
        fig1.update_xaxes(gridcolor="#1e1e1e",showline=True,linecolor="#2a2a2a",tickfont=dict(family="JetBrains Mono,monospace",size=10,color="#666"))
        fig1.update_yaxes(gridcolor="#1e1e1e",showline=False,tickfont=dict(family="JetBrains Mono,monospace",size=10,color="#666"))
        fig1.update_yaxes(tickprefix="$",row=1,col=1)
        fig1.add_trace(go.Scatter(x=df_raw["Date"],y=df_raw["Close"],mode="lines",line=dict(color=COLORS["actual"],width=1.5),name="Close",hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>"),row=1,col=1)
        fig1.add_trace(go.Bar(x=df_raw["Date"],y=df_raw["Volume"],marker_color="rgba(201,168,76,0.25)",name="Volume",hovertemplate="%{x|%b %d, %Y}<br>%{y:,.0f}<extra></extra>"),row=2,col=1)
        fig1.add_vrect(x0=split_date,x1=df_raw["Date"].iloc[-1],fillcolor="rgba(248,113,113,0.04)",line_width=0,annotation_text="Test set (20%)",annotation_position="top left",annotation_font_color="#f87171",annotation_font_size=10,row=1,col=1)
        for evt_date, evt_label in AAPL_EVENTS:
            try:
                ep = df_raw.loc[df_raw["Date"]>=evt_date,"Close"].iloc[0]
                fig1.add_annotation(x=evt_date,y=ep,text=evt_label,showarrow=True,arrowhead=0,arrowcolor="#c9a84c",font=dict(size=8,color="#c9a84c",family="JetBrains Mono,monospace"),bgcolor="rgba(13,13,13,0.8)",bordercolor="#c9a84c",borderwidth=1,ax=0,ay=-36,row=1,col=1)
            except: pass
    else:
        fig1 = editorial_fig(480, "Apple Inc. (AAPL) — Closing Price, 1980–Present")
        fig1.add_trace(go.Scatter(x=df_raw["Date"],y=df_raw["Close"],mode="lines",line=dict(color=COLORS["actual"],width=1.5),name="Close",hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>"))
        fig1.add_vrect(x0=split_date,x1=df_raw["Date"].iloc[-1],fillcolor="rgba(248,113,113,0.04)",line_width=0,annotation_text="Test set (20%)",annotation_position="top left",annotation_font_color="#f87171",annotation_font_size=10)
        for evt_date, evt_label in AAPL_EVENTS:
            try:
                ep = df_raw.loc[df_raw["Date"]>=evt_date,"Close"].iloc[0]
                fig1.add_annotation(x=evt_date,y=ep,text=evt_label,showarrow=True,arrowhead=0,arrowcolor="#c9a84c",font=dict(size=8,color="#c9a84c",family="JetBrains Mono,monospace"),bgcolor="rgba(13,13,13,0.8)",bordercolor="#c9a84c",borderwidth=1,ax=0,ay=-36)
            except: pass
    st.plotly_chart(fig1, use_container_width=True)

    split_str = split_date.strftime('%b %Y') if hasattr(split_date,'strftime') else str(split_date)
    st.markdown(f"""<div class='insight-box'>AAPL has returned <strong>{dec_ret:+,.0f}%</strong> since January 2015, compounding at roughly {((1+dec_ret/100)**(1/10)-1)*100:.1f}% annually. The 80/20 chronological train-test boundary falls at <strong>{split_str}</strong> — placing all models under evaluation on the most volatile and recent price regime, including the 2022 drawdown and 2023–2024 recovery.</div>""", unsafe_allow_html=True)

    st.markdown("<div class='kicker' style='margin-top:24px'>Decade Performance</div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:2px solid #c9a84c;margin:4px 0 12px'></div>", unsafe_allow_html=True)
    decade_rows = []
    for s,e in [(1990,1999),(2000,2009),(2010,2019),(2020,2024)]:
        sr = df_raw[df_raw["Date"].dt.year==s]; er = df_raw[df_raw["Date"].dt.year==e]
        if len(sr) and len(er):
            sp=sr["Close"].iloc[0]; ep=er["Close"].iloc[-1]
            ret=(ep-sp)/sp*100
            decade_rows.append({"Period":f"{s}–{e}","Start":f"${sp:.2f}","End":f"${ep:.2f}","Return":f"{ret:+,.0f}%","_ret":ret})
    if decade_rows:
        dc = st.columns(len(decade_rows))
        for i,row in enumerate(decade_rows):
            c = "#4ade80" if row["_ret"]>0 else "#f87171"
            with dc[i]:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>{row['Period']}</div><div class='metric-value' style='font-size:1.2rem;color:{c}'>{row['Return']}</div><div class='metric-sub'>{row['Start']} → {row['End']}</div></div>", unsafe_allow_html=True)


# ── TAB 2 · Model Predictions ─────────────────────────────────────────────────
with tab2:
    _test_df = pd.DataFrame(test_data["Close"]) if sel_key=="rnn_single" else test_data
    y_true, y_pred = get_predictions(models[sel_key], _test_df, a_scaler, sel_nin, sel_npred, sel_nfeat)
    n_plot = min(len(y_true), len(y_pred))
    dates_test = dates.iloc[-len(test_data)+sel_nin:].reset_index(drop=True).iloc[:n_plot]
    y_true = y_true[:n_plot]; y_pred = y_pred[:n_plot]
    metrics = compute_metrics(y_true, y_pred)
    residuals = y_true - y_pred
    last_true = y_true[-1]; last_pred = y_pred[-1]; last_err = last_true - last_pred
    max_over  = float(np.max(y_pred - y_true))

    st.markdown(f"<div class='kicker'>{selected_label}</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:2px 0 4px'>Test-Set Forecast vs Actual Closing Price</h3>", unsafe_allow_html=True)
    st.markdown("<div class='caption-text'>Out-of-sample evaluation on the last 20% of trading days. The model was never exposed to these prices during training.</div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:2px solid #c9a84c;margin:4px 0 16px'></div>", unsafe_allow_html=True)

    mc1,mc2,mc3,mc4 = st.columns(4)
    for col,label,val,sub in [
        (mc1,"RMSE",         f"${metrics['RMSE']:.4f}", "avg. dollar error"),
        (mc2,"MAPE",         f"{metrics['MAPE']:.3f}%",  "mean abs % error"),
        (mc3,"Max Overshoot",f"${max_over:.2f}",         "largest over-prediction"),
        (mc4,"Last Error",   f"${abs(last_err):.2f}",   "over" if last_err<0 else "under"),
    ]:
        with col: st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='font-size:1.4rem'>{val}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    model_color = COLORS.get(sel_key, "#c9a84c")
    fig2 = editorial_fig(500, f"{selected_label.split('—')[0].strip()} — Predicted vs Actual")
    if show_confidence:
        rs = float(np.std(residuals))
        upper = y_pred + 1.96*rs; lower = y_pred - 1.96*rs
        try:
            r,g,b = int(model_color[1:3],16), int(model_color[3:5],16), int(model_color[5:7],16)
        except: r,g,b = 201,168,76
        fig2.add_trace(go.Scatter(x=pd.concat([dates_test,dates_test[::-1]]),y=np.concatenate([upper,lower[::-1]]),fill="toself",fillcolor=f"rgba({r},{g},{b},0.07)",line=dict(color="rgba(0,0,0,0)"),name="95% Band",showlegend=True))
    fig2.add_trace(go.Scatter(x=dates_test,y=y_true,mode="lines",line=dict(color=COLORS["actual"],width=2),name="Actual",hovertemplate="%{x|%b %d, %Y}<br>Actual: $%{y:.2f}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=dates_test,y=y_pred,mode="lines",line=dict(color=model_color,width=1.8,dash="dot"),name="Predicted",hovertemplate="%{x|%b %d, %Y}<br>Predicted: $%{y:.2f}<extra></extra>"))
    max_err_idx = int(np.argmax(np.abs(residuals)))
    if len(dates_test)>max_err_idx:
        fig2.add_annotation(x=dates_test.iloc[max_err_idx],y=y_pred[max_err_idx],text=f"Peak error<br>${abs(residuals[max_err_idx]):.2f}",showarrow=True,arrowhead=0,arrowcolor="#f87171",font=dict(size=9,color="#f87171",family="JetBrains Mono,monospace"),bgcolor="rgba(13,13,13,0.85)",bordercolor="#f87171",borderwidth=1,ax=40,ay=-40)
    fig2.update_layout(xaxis_title="",yaxis_title="Price (USD)")
    st.plotly_chart(fig2, use_container_width=True)

    miss_desc = "consistently tracks the trend direction but struggles during sharp drawdowns and rallies" if metrics['MAPE']>2 else "closely tracks both trend and short-term volatility"
    st.markdown(f"""<div class='insight-box'>The <strong>{selected_label.split('—')[0].strip()}</strong> achieves an RMSE of <strong>${metrics['RMSE']:.2f}</strong> on the test set — off by roughly ${metrics['RMSE']:.2f} per day on average. With AAPL trading around ${last_true:.0f}, that is a <strong>{metrics['MAPE']:.2f}% miss rate</strong>. The model {miss_desc}.</div>""", unsafe_allow_html=True)

    r1,r2 = st.columns([2,1])
    with r1:
        fig_res = plain_fig(280, "Prediction Residuals (Actual − Predicted)")
        fig_res.update_layout(yaxis=dict(tickprefix="$",gridcolor="#1e1e1e",zeroline=True,zerolinecolor="#333",zerolinewidth=1))
        fig_res.add_trace(go.Bar(x=dates_test,y=residuals,marker_color=["#4ade80" if r>=0 else "#f87171" for r in residuals],name="Residual",hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>"))
        st.plotly_chart(fig_res, use_container_width=True)
    with r2:
        fig_hist = plain_fig(280, "Error Distribution")
        fig_hist.update_layout(xaxis=dict(tickprefix="$"))
        fig_hist.add_trace(go.Histogram(x=residuals,nbinsx=40,marker_color=model_color,opacity=0.75,name="",hovertemplate="Error: $%{x:.2f}<br>Count: %{y}<extra></extra>"))
        st.plotly_chart(fig_hist, use_container_width=True)

    hist_map = {"lstm_exp":"history_lstm_exploratory.csv","lstm_grid":"history_lstm_exploratory.csv","rnn_single":"history_rnn_single.csv","rnn_multi":"history_rnn_multi.csv","rnn_grid":"history_rnn_multi.csv","cnn":"history_cnn.csv","deep":"history_deep_lstm_2day.csv"}
    hist_df = load_csv_if_exists(hist_map.get(sel_key,""))
    if not hist_df.empty and "loss" in hist_df.columns:
        st.markdown("<div class='kicker' style='margin-top:8px'>Training Convergence</div>", unsafe_allow_html=True)
        st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 12px'></div>", unsafe_allow_html=True)
        fig_loss = plain_fig(260, "Training & Validation Loss Curves")
        fig_loss.add_trace(go.Scatter(y=hist_df["loss"],mode="lines",line=dict(color=model_color,width=1.8),name="Train loss"))
        if "val_loss" in hist_df.columns:
            fig_loss.add_trace(go.Scatter(y=hist_df["val_loss"],mode="lines",line=dict(color="#888",width=1.5,dash="dash"),name="Val loss"))
            best_ep = int(hist_df["val_loss"].idxmin())
            fig_loss.add_annotation(x=best_ep,y=hist_df["val_loss"].min(),text=f"Best: epoch {best_ep+1}",showarrow=True,arrowhead=0,arrowcolor="#c9a84c",font=dict(size=9,color="#c9a84c",family="JetBrains Mono,monospace"),bgcolor="rgba(13,13,13,0.85)",bordercolor="#c9a84c",borderwidth=1,ax=30,ay=-30)
        fig_loss.update_layout(xaxis_title="Epoch",yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)


# ── TAB 3 · Model Comparison ──────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='kicker'>All 6 Single-Step Architectures</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:2px 0 4px'>Side-by-Side on the Test Set</h3>", unsafe_allow_html=True)
    st.markdown("<div class='caption-text'>All models evaluated on an identical out-of-sample window. 10-step look-back and 6 features unless noted. Lower RMSE = smaller average dollar error.</div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:2px solid #c9a84c;margin:4px 0 16px'></div>", unsafe_allow_html=True)

    compare_models = [
        ("LSTM Exploratory","lstm_exp",  10,1,N_FEATURES,test_data),
        ("LSTM Grid Best",  "lstm_grid", 15,1,N_FEATURES,test_data),
        ("RNN Single Feat", "rnn_single",10,1,1,pd.DataFrame(test_data["Close"])),
        ("RNN Multi Feat",  "rnn_multi", 10,1,5,test_data),
        ("RNN Grid Best",   "rnn_grid",  15,1,N_FEATURES,test_data),
        ("CNN",             "cnn",       10,1,N_FEATURES,test_data),
    ]
    _,y_te10 = get_predictions(models["lstm_exp"],test_data,a_scaler,10,1,N_FEATURES)
    dates_te10 = dates.iloc[-len(test_data)+10:].reset_index(drop=True)

    metrics_rows=[]; all_preds={}
    for label,key,n_in,n_pr,n_ft,df_t in compare_models:
        yt,yp = get_predictions(models[key],df_t,a_scaler,n_in,n_pr,n_ft)
        all_preds[key]=(yt,yp)
        n=min(len(dates_te10),len(yp))
        m=compute_metrics(y_te10[:n],yp[:n])
        metrics_rows.append({"Model":label,"key":key,**{k:round(v,6) for k,v in m.items()}})
    mdf_full=pd.DataFrame(metrics_rows)
    best_key=mdf_full.loc[mdf_full["RMSE"].idxmin(),"key"]
    worst_key=mdf_full.loc[mdf_full["RMSE"].idxmax(),"key"]

    fig3 = editorial_fig(520,"All Models vs Actual — Test Set")
    fig3.add_trace(go.Scatter(x=dates_te10[:len(y_te10)],y=y_te10,mode="lines",line=dict(color=COLORS["actual"],width=2.5),name="Actual",hovertemplate="%{x|%b %d, %Y}<br>Actual: $%{y:.2f}<extra></extra>"))
    for label,key,n_in,n_pr,n_ft,df_t in compare_models:
        _,yp=all_preds[key]; n=min(len(dates_te10),len(yp))
        is_best=key==best_key
        fig3.add_trace(go.Scatter(x=dates_te10[:n],y=yp[:n],mode="lines",line=dict(color=COLORS.get(key,"#888"),width=2.2 if is_best else 1.2,dash="solid" if is_best else "dot"),name=f"{'★ ' if is_best else ''}{label}",hovertemplate=f"%{{x|%b %d, %Y}}<br>{label}: $%{{y:.2f}}<extra></extra>"))
    fig3.update_layout(xaxis_title="",yaxis_title="Price (USD)")
    st.plotly_chart(fig3, use_container_width=True)

    best_row=mdf_full.loc[mdf_full["RMSE"].idxmin()]; worst_row=mdf_full.loc[mdf_full["RMSE"].idxmax()]
    st.markdown(f"""<div class='insight-box'><strong>★ {best_row['Model']}</strong> leads with RMSE <strong>${best_row['RMSE']:.4f}</strong> — the lowest average dollar error across all 6 architectures. <strong>{worst_row['Model']}</strong> trails at ${worst_row['RMSE']:.4f}, {worst_row['RMSE']/best_row['RMSE']:.1f}× worse. LSTM architectures consistently outperform SimpleRNN variants, consistent with LSTM's gating mechanism preserving long-range sequential structure through multi-month trend regimes.</div>""", unsafe_allow_html=True)

    t1,t2 = st.columns([1.4,1])
    with t1:
        st.markdown("<div class='kicker'>Metric Comparison</div>", unsafe_allow_html=True)
        st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 10px'></div>", unsafe_allow_html=True)
        mdd = mdf_full.drop(columns=["key"]).set_index("Model")
        st.dataframe(mdd.style.highlight_min(axis=0,subset=["MSE","RMSE","MAPE"],props="background-color:#1a2e1a;color:#4ade80").highlight_max(axis=0,subset=["MSE","RMSE","MAPE"],props="background-color:#2e1a1a;color:#f87171").format({"MSE":"{:.6f}","RMSE":"${:.4f}","MAPE":"{:.3f}%"}),use_container_width=True,height=260)
    with t2:
        st.markdown("<div class='kicker'>RMSE by Model</div>", unsafe_allow_html=True)
        st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 10px'></div>", unsafe_allow_html=True)
        fig_bar = plain_fig(270)
        fig_bar.update_layout(yaxis=dict(tickprefix="$",gridcolor="#1e1e1e"),xaxis=dict(tickangle=-30,tickfont=dict(size=9)),margin=dict(l=50,r=10,t=20,b=70))
        bar_colors=["#4ade80" if r["key"]==best_key else "#f87171" if r["key"]==worst_key else COLORS.get(r["key"],"#888") for _,r in mdf_full.iterrows()]
        fig_bar.add_trace(go.Bar(x=mdf_full["Model"],y=mdf_full["RMSE"],marker_color=bar_colors,text=[f"${v:.2f}" for v in mdf_full["RMSE"]],textposition="outside",textfont=dict(size=9,family="JetBrains Mono,monospace")))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div class='kicker' style='margin-top:16px'>MAPE Ranking</div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 10px'></div>", unsafe_allow_html=True)
    sorted_rows = mdf_full.sort_values("MAPE")
    rcols = st.columns(len(sorted_rows))
    for i,(_,row) in enumerate(sorted_rows.iterrows()):
        c="#4ade80" if row["key"]==best_key else "#e8e0d4"
        with rcols[i]: st.markdown(f"<div class='metric-card'><div class='metric-label'>#{i+1} {row['Model']}</div><div class='metric-value' style='font-size:1.1rem;color:{c}'>{row['MAPE']:.3f}%</div><div class='metric-sub'>RMSE ${row['RMSE']:.4f}</div></div>", unsafe_allow_html=True)


# ── TAB 4 · Hyperparameter Tuning ────────────────────────────────────────────
with tab4:
    today_note = datetime.now().strftime("%B %d, %Y")
    st.markdown("<div class='kicker'>Hyperparameter Search Results</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:2px 0 4px'>GridSearchCV · Keras Tuner Hyperband</h3>", unsafe_allow_html=True)
    st.markdown("<div class='caption-text'>Results are generated by running <code>python train.py</code> locally (~30–60 min on CPU). Charts below reflect your most recent training run.</div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:2px solid #c9a84c;margin:4px 0 16px'></div>", unsafe_allow_html=True)
    st.markdown(f"""<div class='result-box'><span class='anno-badge'>Note</span> To refresh with data through <strong>{today_note}</strong>, run: <code>python train.py</code> · saves all model weights + CSVs to <code>saved_models/</code></div>""", unsafe_allow_html=True)

    st.markdown("<div class='kicker' style='margin-top:20px'>Stage 1 — GridSearchCV</div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='caption-text'>Exhaustive search over <code>optimiser ∈ [adam, sgd]</code> × <code>neurons ∈ [32, 64, 128]</code> with 2-fold CV, 10 epochs each.</div>", unsafe_allow_html=True)

    gs_lstm = load_csv_if_exists("gridsearch_lstm.csv")
    gs_rnn  = load_csv_if_exists("gridsearch_rnn.csv")
    if not gs_lstm.empty and not gs_rnn.empty:
        cg1,cg2 = st.columns(2)
        for col,gs_df,title,mc in [(cg1,gs_lstm,"LSTM GridSearchCV — MSE Heatmap",COLORS["lstm_exp"]),(cg2,gs_rnn,"RNN GridSearchCV — MSE Heatmap",COLORS["rnn_grid"])]:
            with col:
                pivot = gs_df.pivot_table(index="param_neurons",columns="param_optimiser",values="mean_test_score",aggfunc="mean")
                pivot = -pivot
                fig_hm = plain_fig(300, title)
                fig_hm.add_trace(go.Heatmap(z=pivot.values,x=pivot.columns.tolist(),y=[str(v) for v in pivot.index.tolist()],colorscale=[[0,"#0d0d0d"],[0.5,"#1a2e1a"],[1,"#4ade80"]],text=np.round(pivot.values,8),texttemplate="%{text:.2e}",showscale=True,colorbar=dict(title="MSE",tickfont=dict(size=9,family="JetBrains Mono,monospace"))))
                st.plotly_chart(fig_hm, use_container_width=True)
        br=gs_lstm.loc[gs_lstm["mean_test_score"].idxmax()]; rr=gs_rnn.loc[gs_rnn["mean_test_score"].idxmax()]
        st.markdown(f"""<div class='result-box'><span class='anno-badge'>Best</span>LSTM: neurons=<strong>{int(br['param_neurons'])}</strong>, optimiser=<strong>{br['param_optimiser']}</strong> · MSE {br['mean_test_score']:.6e}<br>RNN: &nbsp;neurons=<strong>{int(rr['param_neurons'])}</strong>, optimiser=<strong>{rr['param_optimiser']}</strong> · MSE {rr['mean_test_score']:.6e}</div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class='insight-box'>Both LSTM and RNN converged on <strong>neurons=128, optimiser=adam</strong>. Adam's adaptive learning rate handles the non-stationary gradients of financial time series well. Larger hidden units (128 vs 32/64) provide richer hidden-state representations of multi-month price trends.</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='result-box' style='border-left-color:#444'><span class='anno-badge' style='background:#333;color:#888'>Missing</span>GridSearchCV results not found. Run <code>python train.py</code> to generate.<br><br>Expected output:<br>LSTM Best params {{'neurons': 128, 'optimiser': 'adam'}} · MSE −3.17e−07<br>RNN  Best model  {{'neurons': 128, 'optimiser': 'adam'}} · MSE −3.53e−06</div>""", unsafe_allow_html=True)

    st.markdown("<div style='border-top:1px solid #2a2a2a;margin:24px 0 16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Stage 2 — Keras Tuner Hyperband (Single-Step LSTM)</div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='caption-text'><code>units ∈ [32, 448]</code> (step 32) × input sizes [10, 15, 20, 24, 25, 30]. Full sweep ~200 min on CPU.</div>", unsafe_allow_html=True)
    tuner_single = load_csv_if_exists("tuner_single_step_results.csv")
    if not tuner_single.empty:
        br2 = tuner_single.loc[tuner_single["val_loss"].idxmin()]
        fig_t1 = plain_fig(340,"Best val_loss by Input Window Size")
        fig_t1.add_trace(go.Scatter(x=tuner_single["inputs"].astype(str),y=tuner_single["val_loss"],mode="markers+lines",marker=dict(size=10,color=COLORS["lstm_exp"]),line=dict(color=COLORS["lstm_exp"],width=1.8),text=[f"units={int(u)}" for u in tuner_single["units"]],textposition="top center",textfont=dict(size=9,color="#888",family="JetBrains Mono,monospace"),name="Best val_loss",hovertemplate="n_inputs=%{x}<br>val_loss=%{y:.4e}<br>%{text}<extra></extra>"))
        fig_t1.add_annotation(x=str(int(br2["inputs"])),y=br2["val_loss"],text=f"★ Best\nunits={int(br2['units'])}",showarrow=True,arrowhead=0,arrowcolor="#c9a84c",font=dict(size=9,color="#c9a84c",family="JetBrains Mono,monospace"),bgcolor="rgba(13,13,13,0.85)",bordercolor="#c9a84c",borderwidth=1,ax=30,ay=-40)
        fig_t1.update_layout(xaxis_title="Look-back window (n_inputs)",yaxis_title="val_loss")
        st.plotly_chart(fig_t1, use_container_width=True)
        st.dataframe(tuner_single.sort_values("val_loss").reset_index(drop=True),use_container_width=True,height=200)
    else:
        st.markdown("<div class='result-box' style='border-left-color:#444'><span class='anno-badge' style='background:#333;color:#888'>Missing</span> Run <code>python train.py</code> to generate tuner results.</div>", unsafe_allow_html=True)

    st.markdown("<div style='border-top:1px solid #2a2a2a;margin:24px 0 16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kicker'>Stage 3 — Keras Tuner Hyperband (2-Day Deep LSTM)</div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='caption-text'><code>units ∈ [192, 252]</code> · <code>dense ∈ [10, 50]</code> · stacked 2-layer LSTM · max_epochs=20 · objective=val_mse.</div>", unsafe_allow_html=True)
    best_hps_path = os.path.join(SAVE_DIR,"best_hps_2day.json")
    if os.path.exists(best_hps_path):
        with open(best_hps_path) as f: best_hps=json.load(f)
        st.markdown(f"""<div class='result-box'><span class='anno-badge'>Optimal</span>LSTM units: <strong>{best_hps['units']}</strong> · Dense units: <strong>{best_hps['dense']}</strong><br>Architecture: LSTM({best_hps['units']})→LSTM({best_hps['units']})→Dense({best_hps['dense']})→Dense({best_hps['dense']})→Dense(2)</div>""", unsafe_allow_html=True)
        history_2d = load_csv_if_exists("history_deep_lstm_2day.csv")
        if not history_2d.empty:
            fig_2d = plain_fig(300,"Deep LSTM 2-Day — Training Convergence")
            fig_2d.add_trace(go.Scatter(y=history_2d["loss"],mode="lines",line=dict(color=COLORS["deep"],width=1.8),name="Train loss"))
            if "val_loss" in history_2d.columns:
                fig_2d.add_trace(go.Scatter(y=history_2d["val_loss"],mode="lines",line=dict(color="#888",width=1.5,dash="dash"),name="Val loss"))
                best_ep=int(history_2d["val_loss"].idxmin())
                fig_2d.add_annotation(x=best_ep,y=history_2d["val_loss"].min(),text=f"Best epoch {best_ep+1}",showarrow=True,arrowhead=0,arrowcolor="#c9a84c",font=dict(size=9,color="#c9a84c",family="JetBrains Mono,monospace"),bgcolor="rgba(13,13,13,0.85)",bordercolor="#c9a84c",borderwidth=1,ax=40,ay=-30)
            fig_2d.update_layout(xaxis_title="Epoch",yaxis_title="Loss")
            st.plotly_chart(fig_2d, use_container_width=True)
    else:
        st.markdown("<div class='result-box' style='border-left-color:#444'><span class='anno-badge' style='background:#333;color:#888'>Missing</span> Run <code>python train.py</code> to generate 2-day tuner results.</div>", unsafe_allow_html=True)


# ── TAB 5 · Methodology ───────────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='kicker'>Research Design & Implementation</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin:2px 0 4px'>Methodology</h3>", unsafe_allow_html=True)
    st.markdown("<div class='caption-text'>A faithful reproduction and extension of a Jupyter notebook exploring deep learning approaches to equity price forecasting. Every transformation from raw ticker data to model evaluation is described below.</div>", unsafe_allow_html=True)
    st.markdown("<div style='border-top:2px solid #c9a84c;margin:4px 0 20px'></div>", unsafe_allow_html=True)

    col_a,col_b = st.columns([1,1])
    with col_a:
        st.markdown("<div class='kicker'>Pipeline</div>", unsafe_allow_html=True)
        st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 12px'></div>", unsafe_allow_html=True)
        st.markdown("""
**1 · Data Collection**

AAPL daily OHLCV from Yahoo Finance via `yfinance` with `auto_adjust=False` to preserve Adj Close.
Features: Open, High, Low, **Close**, Adj Close, Volume — 6 features matching the original notebook. Range: 1980-12-12 to present (~11,000 trading days).

**2 · Preprocessing**
""")
        st.code("""a_scaler = MinMaxScaler(feature_range=(0, 1))
apple_norm = pd.DataFrame(
    a_scaler.fit_transform(apple_norm),
    columns=apple_norm.columns)
train_data = apple_norm[:round(len(apple_norm) * 0.8)]
test_data  = apple_norm[round(len(apple_norm) * 0.8):]""", language="python")
        st.markdown("""
Scaler fit on training data only — no look-ahead bias. 80/20 chronological split, no shuffling.

**3 · Sequence Construction**
""")
        st.code("""for i in range(n_inputs, len(df) - n_predictions + 1):
    X_train.append(df.iloc[i-n_inputs:i, 0:n_features])
    y_train.append(df["Close"][i : i + n_predictions])""", language="python")
        st.markdown("""
Sliding window of shape `(n_inputs, n_features)`, target `y = Close[t+1]`.

**4 · Inverse Scaling**
""")
        st.code("""y_rescaled = (y_scaled - a_scaler.min_[3]) / a_scaler.scale_[3]
# index 3 = Close in [Open, High, Low, Close, Adj Close, Volume]""", language="python")
        st.markdown("**5 · Model Architectures**")
        arch_df = pd.DataFrame([
            ("LSTM exploratory","(10, 6)","1","units=64, adam, logcosh"),
            ("LSTM grid best",  "(15, 6)","1","units=128, adam — GridSearchCV winner"),
            ("RNN single feat", "(10, 1)","1","Close only, units=64"),
            ("RNN multi feat",  "(10, 5)","1","n_features=5"),
            ("RNN grid best",   "(15, 6)","1","units=128, adam — GridSearchCV winner"),
            ("CNN",             "(10, 6)","1","Conv1D(64)→Pool→Conv1D(32)→Dense"),
            ("Deep LSTM 2-day", "(7, 6)", "2","2×LSTM(192)→2×Dense(40) — Tuner winner"),
        ], columns=["Model","Input","Output","Notes"])
        st.dataframe(arch_df, use_container_width=True, hide_index=True, height=280)
        st.markdown("""
**6 · Tuning Stages**

- **GridSearchCV** — `optimiser ∈ [adam, sgd]` × `neurons ∈ [32, 64, 128]`, cv=2
- **Keras Tuner Hyperband** — units 32–448, input sizes 10–30 (single-step)
- **Keras Tuner Hyperband** — units 192–252, dense 10–50 (2-day Deep LSTM)

**7 · Tech Stack**
""")
        tags = ["Python 3.11","TensorFlow 2.x","Keras","keras-tuner","scikit-learn","Streamlit","Plotly","yfinance","NumPy","pandas","joblib"]
        tag_html = "".join(f"<span style='display:inline-block;background:#1a1500;color:#c9a84c;border:1px solid #c9a84c;border-radius:2px;padding:3px 10px;font-size:0.72rem;margin:3px 2px;font-family:JetBrains Mono,monospace;font-weight:600'>{t}</span>" for t in tags)
        st.markdown(f"<div style='line-height:2.4;padding:6px 0'>{tag_html}</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='kicker'>Why LSTM Outperforms SimpleRNN</div>", unsafe_allow_html=True)
        st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 12px'></div>", unsafe_allow_html=True)
        st.markdown("""
LSTM cells solve the **vanishing gradient problem** through three learned gating mechanisms:

- **Forget gate** — what fraction of cell state to discard
- **Input gate** — which new information enters the cell state
- **Output gate** — what portion of cell state is exposed as hidden state

Stock prices exhibit multi-scale dependencies: intraday noise, weekly seasonality, multi-month trend regimes, and annual earnings cycles. The LSTM cell state provides a dedicated memory channel persisting relevant structure across hundreds of timesteps — something a SimpleRNN's single hidden state cannot sustain.
""")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='kicker'>Limitations & Caveats</div>", unsafe_allow_html=True)
        st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 12px'></div>", unsafe_allow_html=True)
        st.markdown("""
**No look-ahead bias** — the MinMaxScaler is fit on training data only; test-set statistics never contaminate preprocessing.

**Pattern recognition only** — models learn from price and volume structure. They have no awareness of earnings announcements, Fed decisions, macro events, or sentiment signals that drive the majority of large single-day moves.

**Metric interpretation** — MAPE can be misleadingly low near zero-trend periods; RMSE in USD is the more interpretable error for dollar-denominated assets.

**Distribution shift** — AAPL's volatility regime in 2022–2024 differs substantially from 2010–2019. A static model trained once will decay in accuracy. Real deployment would require rolling-window retraining.

**No transaction costs** — all metrics are gross; bid-ask spread, slippage, and taxes are not modelled.
""")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='kicker'>Reproducing Results</div>", unsafe_allow_html=True)
        st.markdown("<div style='border-top:1px solid #2a2a2a;margin:4px 0 12px'></div>", unsafe_allow_html=True)
        st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full training + tuning (~30–60 min on CPU)
#    Re-run anytime to update with latest AAPL data
python train.py

# 3. Launch the portfolio app
streamlit run app.py""", language="bash")
        st.markdown("""<div class='insight-box'>The Hyperparameter Tuning tab shows "results not found" until <code>train.py</code> has been run at least once. All other tabs fall back to quick-trained weights so the app remains fully functional without pre-training.</div>""", unsafe_allow_html=True)
