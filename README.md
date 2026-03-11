---
title: Apple Stock LSTM Deep Learning
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# Apple Stock · Deep Learning Forecast

An interactive Streamlit portfolio app that faithfully reproduces every
experiment from a deep learning stock prediction notebook.

## What's inside

| File | Purpose |
|------|---------|
| `data_loader.py` | yfinance data fetch, preprocessing, `preprocess_lstm()`, `rescale_close()` — all verbatim from notebook |
| `models.py` | All 5 architectures + Keras Tuner builder functions |
| `train.py` | Full training pipeline: exploratory fits → GridSearchCV → Keras Tuner Hyperband (single-step) → Keras Tuner Hyperband (2-day) → CNN |
| `app.py` | 5-tab Streamlit dashboard with live predictions, tuning heatmaps, loss curves |

## Models

- **LSTM exploratory** — (10-step, 6 features)
- **LSTM grid best** — GridSearchCV winner (neurons=128, adam)
- **RNN single feature** — Close price only
- **RNN multi feature** — 5 features
- **RNN grid best** — GridSearchCV winner
- **1-D CNN** — Conv1D(64,k=3) → MaxPool → Conv1D(32,k=3) → Dense(128) → Dropout(0.15) → Dense
- **Deep LSTM 2-day** — Keras Tuner best: 2×LSTM(192) → 2×Dense(40) → Dense(2)

## Hyperparameter Tuning

1. **GridSearchCV** — `optimiser=['adam','sgd'] × neurons=[32,64,128]`, cv=2
2. **Keras Tuner Hyperband** — units 32–448 across input sizes 10–30 (single-step)
3. **Keras Tuner Hyperband** — units 192–252, dense 10–50 (2-day model)

## Run locally

```bash
pip install -r requirements.txt

# One-time: train all models + run all tuning (~30–60 min on CPU)
python train.py

# Launch app
streamlit run app.py
```

## Deploy to Hugging Face Spaces

1. Create a new Space (SDK: Streamlit)
2. Add `HF_TOKEN` to your GitHub repo secrets
3. Update `YOUR_HF_USERNAME` in `.github/workflows/deploy.yml`
4. Push to `main` — GitHub Actions auto-deploys

## Disclaimer

Educational project only. Not financial advice.
