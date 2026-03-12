"""
data_loader.py
==============
Fetches and preprocesses Apple (AAPL) stock data using yfinance.

Original notebook used a CSV with columns:
    Date | Open | High | Low | Close | Adj Close | Volume  (6 numeric features)

We replicate that exactly.  yfinance returns Adj Close when auto_adjust=False.

Rescaling formula taken verbatim from the notebook:
    y_rescaled = (y_scaled - a_scaler.min_[3]) / a_scaler.scale_[3]
where index 3 = position of 'Close' in [Open, High, Low, Close, Adj Close, Volume]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Column order must match the original CSV exactly — 6 numeric features
FEATURE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
N_FEATURES   = 6      # kept as module-level constant
CLOSE_IDX    = 3      # index of 'Close' inside FEATURE_COLS


# ── Data acquisition ──────────────────────────────────────────────────────────

def fetch_aapl_data(start: str = "1980-01-01", end: str = None) -> pd.DataFrame:
    """
    Download full AAPL OHLCV + Adj Close history from Yahoo Finance.
    Tries multiple approaches with a short timeout so it fails fast
    in restricted environments (e.g. Hugging Face Spaces).
    """
    # Let yfinance manage its own session (required for newer versions)
    df = yf.download(
        "AAPL",
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df is None or len(df) == 0:
        raise ValueError("yfinance returned empty data for AAPL")

    df = df.reset_index()

    # Flatten MultiIndex columns if present (newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date"] + FEATURE_COLS].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df


def load_from_csv(path: str) -> pd.DataFrame:
    """Load the bundled AAPL.csv (offline fallback)."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date"] + FEATURE_COLS].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Replicate the notebook's preprocessing block verbatim:

        apple_norm = aapl.copy()
        dates = apple_norm['Date']
        apple_norm = apple_norm.drop(['Date'], axis=1)
        a_scaler = MinMaxScaler(feature_range=(0, 1))
        apple_norm = pd.DataFrame(a_scaler.fit_transform(apple_norm),
                                  columns=apple_norm.columns)
        train_data = apple_norm[:round(len(apple_norm["Open"])*0.8)]
        test_data  = apple_norm[round(len(apple_norm["Open"])*0.8):]

    Returns
    -------
    train_data : pd.DataFrame  scaled, no Date column
    test_data  : pd.DataFrame  scaled, no Date column
    a_scaler   : fitted MinMaxScaler  (named a_scaler to match notebook)
    dates      : pd.Series of original datetime values
    """
    apple_norm = df.copy()
    dates      = pd.to_datetime(apple_norm["Date"]).reset_index(drop=True)

    apple_norm = apple_norm.drop(columns=["Date"])

    a_scaler   = MinMaxScaler(feature_range=(0, 1))
    apple_norm = pd.DataFrame(
        a_scaler.fit_transform(apple_norm),
        columns=apple_norm.columns
    )

    split      = round(len(apple_norm["Open"]) * train_ratio)
    train_data = apple_norm.iloc[:split].reset_index(drop=True)
    test_data  = apple_norm.iloc[split:].reset_index(drop=True)

    return train_data, test_data, a_scaler, dates


# ── Sequence construction — verbatim notebook function ────────────────────────

def preprocess_lstm(df: pd.DataFrame,
                    n_inputs: int      = 10,
                    n_predictions: int = 1,
                    n_features: int    = 6):
    """
    Exact copy of the notebook's preprocess_lstm() helper:

        def preprocess_lstm(df, n_inputs=10, n_predictions=1, n_features=6):
            X_train, y_train = [], []
            for i in range(n_inputs, len(df) - n_predictions + 1):
                X_train.append(df.iloc[i-n_inputs:i , 0:n_features])
                y_train.append(df["Close"][i : i + n_predictions])
            X_train, y_train = np.array(X_train), np.array(y_train)
            return X_train, y_train

    Note: the notebook uses df["Close"][i : i + n_predictions] (index-based
    access on the Series) which is equivalent to .iloc.  We use .iloc for
    safety.
    """
    X_train, y_train = [], []

    for i in range(n_inputs, len(df) - n_predictions + 1):
        X_train.append(df.iloc[i - n_inputs:i, 0:n_features].values)
        y_train.append(df["Close"].iloc[i: i + n_predictions].values)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    return X_train, y_train


# ── Inverse scaling — verbatim notebook formula ───────────────────────────────

def rescale_close(arr: np.ndarray, a_scaler: MinMaxScaler) -> np.ndarray:
    """
    Inverse-transform normalised Close predictions back to dollar prices.

    Taken directly from every prediction block in the notebook:
        y_testing   = (y_testing   - a_scaler.min_[3]) / a_scaler.scale_[3]
        y_predicted = (y_predicted - a_scaler.min_[3]) / a_scaler.scale_[3]

    MinMaxScaler forward transform:  X_scaled = X * scale_ + min_
    Inverse:                         X = (X_scaled - min_) / scale_

    CLOSE_IDX = 3  →  'Close' is the 4th column in
    [Open(0), High(1), Low(2), Close(3), Adj Close(4), Volume(5)]
    """
    return (arr - a_scaler.min_[CLOSE_IDX]) / a_scaler.scale_[CLOSE_IDX]
