"""
train.py
========
Full training pipeline replicating every experiment from the notebook:

  1. Exploratory LSTM   — 10-step look-back, 1-step prediction, 6 features
  2. SimpleRNN single   — 10-step, 1-step, Close price only (1 feature)
  3. SimpleRNN multi    — 10-step, 1-step, 5 features (notebook used n_features=5)
  4. GridSearchCV       — optimiser × neurons for both LSTM and RNN
                          optimiser = ['adam', 'sgd']
                          neurons   = [32, 64, 128]
                          (full set also includes adagrad/adadelta/rmsprop
                          but the notebook comments those out for runtime)
  5. Keras Tuner        — Hyperband on single-step LSTM over input sizes
  6. Deep LSTM          — kt.Hyperband for 2-day prediction (7-step look-back)
  7. 1-D CNN            — architecture from notebook, 10-step / 6-features

All models, scaler, and tuning results are saved to ./saved_models/.

Runtime note: GridSearchCV (step 4) takes ~10–15 min on CPU.
              Keras Tuner (step 5) takes ~5–10 min on CPU.
              Deep LSTM Tuner (step 6) takes ~10–20 min on CPU.

Run once before launching the Streamlit app:
    python train.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import joblib

from data_loader import fetch_aapl_data, preprocess, preprocess_lstm, rescale_close, N_FEATURES
from models import (
    build_lstm, build_rnn_single, build_rnn_multi, build_cnn, build_deep_lstm,
    model_builder_single_step, model_builder_2day
)

# ── Reproducibility (match notebook) ─────────────────────────────────────────
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hyperparameters (match notebook) ─────────────────────────────────────────
n_inputs     = 10
n_predictions = 1
batch_size   = 32

# ── Data ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("📥  Fetching AAPL data …")
try:
    aapl = fetch_aapl_data()
except Exception as e:
    print(f"    yfinance failed ({e}) – loading bundled CSV")
    from data_loader import load_from_csv
    aapl = load_from_csv("AAPL.csv")

print(f"    {len(aapl):,} trading days  "
      f"({aapl['Date'].iloc[0].date()} → {aapl['Date'].iloc[-1].date()})")

train_data, test_data, a_scaler, dates = preprocess(aapl)

# Save scaler so the app can inverse-transform without retraining
joblib.dump(a_scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
print("✅  Scaler saved.")

# ── Common early-stopping callback ───────────────────────────────────────────
def es(patience: int = 15, monitor: str = "val_loss"):
    return EarlyStopping(monitor=monitor, patience=patience,
                         restore_best_weights=True, verbose=0)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Exploratory LSTM  (10-step, 6-feature, 1-step prediction)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🧠  [1/7]  Exploratory LSTM (n_inputs=10, n_features=6) …")

X_train, y_train = preprocess_lstm(train_data, n_inputs=10,
                                    n_predictions=1, n_features=N_FEATURES)
print(f"    X_train shape: {X_train.shape}   y_train shape: {y_train.shape}")

model_lstm = build_lstm(optimiser="adam", neurons=64,
                         n_inputs=10, n_features=N_FEATURES)
model_lstm.summary()

history_lstm = model_lstm.fit(
    X_train, y_train,
    epochs=100, batch_size=batch_size,
    validation_split=0.2,
    callbacks=[es(15)],
    verbose=0
)
model_lstm.save(os.path.join(SAVE_DIR, "lstm_exploratory.keras"))
print("    Saved  lstm_exploratory.keras")

# Test predictions
X_test, y_test = preprocess_lstm(test_data, n_inputs=10,
                                   n_predictions=1, n_features=N_FEATURES)
y_pred_lstm = model_lstm.predict(X_test, verbose=0)
y_test_r    = rescale_close(y_test.flatten(), a_scaler)
y_pred_r    = rescale_close(y_pred_lstm.flatten(), a_scaler)

lstm_mse  = float(np.mean((y_test_r - y_pred_r) ** 2))
print(f"    Test MSE (rescaled): {lstm_mse:.4f}")

# Save training history
pd.DataFrame(history_lstm.history).to_csv(
    os.path.join(SAVE_DIR, "history_lstm_exploratory.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SimpleRNN — single feature (Close only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🧠  [2/7]  SimpleRNN — single feature (Close only) …")

X_train_rnn1, y_train_rnn1 = preprocess_lstm(
    pd.DataFrame(train_data["Close"]), n_inputs=10, n_predictions=1, n_features=1)

model_rnn_single = build_rnn_single(optimiser="adam", neurons=64,
                                     n_inputs=10, n_features=1)
model_rnn_single.summary()

history_rnn1 = model_rnn_single.fit(
    X_train_rnn1, y_train_rnn1,
    epochs=100, batch_size=100,
    validation_split=0.2,
    callbacks=[es(5)],
    verbose=0
)
model_rnn_single.save(os.path.join(SAVE_DIR, "rnn_single_feature.keras"))
print("    Saved  rnn_single_feature.keras")

X_test_rnn1, y_test_rnn1 = preprocess_lstm(
    pd.DataFrame(test_data["Close"]), n_inputs=10, n_predictions=1, n_features=1)
y_pred_rnn1 = model_rnn_single.predict(X_test_rnn1, verbose=0)

pd.DataFrame(history_rnn1.history).to_csv(
    os.path.join(SAVE_DIR, "history_rnn_single.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SimpleRNN — multi-feature (5 features as in notebook section B)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🧠  [3/7]  SimpleRNN — multi-feature (n_features=5) …")

n_features_rnn_multi = 5   # notebook uses n_features=5 here explicitly

X_train_rnnm, y_train_rnnm = preprocess_lstm(
    train_data, n_inputs=10, n_predictions=1, n_features=n_features_rnn_multi)

model_rnn_multi = build_rnn_multi(optimiser="adam", neurons=64,
                                   n_inputs=10, n_features=n_features_rnn_multi)
model_rnn_multi.summary()

history_rnnm = model_rnn_multi.fit(
    X_train_rnnm, y_train_rnnm,
    epochs=15, batch_size=100,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3,
                              restore_best_weights=True, verbose=0)],
    verbose=0
)
# Second fit pass (notebook does this explicitly)
history_rnnm2 = model_rnn_multi.fit(
    X_train_rnnm, y_train_rnnm,
    epochs=15, batch_size=100,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3,
                              restore_best_weights=True, verbose=0)],
    verbose=0
)
model_rnn_multi.save(os.path.join(SAVE_DIR, "rnn_multi_feature.keras"))
print("    Saved  rnn_multi_feature.keras")

pd.DataFrame(history_rnnm2.history).to_csv(
    os.path.join(SAVE_DIR, "history_rnn_multi.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GridSearchCV — LSTM and RNN
#     optimiser × neurons, cv=2, 10 epochs each fold
#     Input shape uses n_inputs=15 (as in notebook grid-search data prep)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🔍  [4/7]  GridSearchCV — LSTM & RNN …")
print("    (optimiser=[adam, sgd], neurons=[32, 64, 128], cv=2, epochs=10)")
print("    This will take ~10–15 minutes on CPU …")

# Notebook uses n_inputs=15 for the grid search data
X_train_gs, y_train_gs = preprocess_lstm(
    train_data, n_inputs=15, n_predictions=1, n_features=N_FEATURES)
print(f"    Grid-search X_train shape: {X_train_gs.shape}")

optimiser_options = ["adam", "sgd"]
neuron_options    = [32, 64, 128]

# --- Manual 2-fold cross-validation (identical to GridSearchCV cv=2) ---
# Replaces KerasRegressor + GridSearchCV which has scikit-learn version
# incompatibilities. Logic is exactly the same: iterate optimiser x neurons,
# split data into 2 folds, train for 10 epochs, record mean MSE.
from sklearn.model_selection import KFold

gs_lstm_rows, gs_rnn_rows = [], []
kf = KFold(n_splits=2, shuffle=False)

for optimiser in optimiser_options:
    for neurons in neuron_options:
        print(f"    Testing optimiser={optimiser}, neurons={neurons} …")
        lstm_scores, rnn_scores = [], []

        for tr_idx, val_idx in kf.split(X_train_gs):
            X_tr, X_val = X_train_gs[tr_idx], X_train_gs[val_idx]
            y_tr, y_val = y_train_gs[tr_idx], y_train_gs[val_idx]

            m_lstm = build_lstm(optimiser=optimiser, neurons=neurons,
                                n_inputs=15, n_features=N_FEATURES)
            m_lstm.fit(X_tr, y_tr, epochs=10, batch_size=100, verbose=0)
            pred = m_lstm.predict(X_val, verbose=0)
            lstm_scores.append(-float(np.mean((pred.flatten() - y_val.flatten()) ** 2)))

            m_rnn = build_rnn_multi(optimiser=optimiser, neurons=neurons,
                                    n_inputs=15, n_features=N_FEATURES)
            m_rnn.fit(X_tr, y_tr, epochs=10, batch_size=100, verbose=0)
            pred = m_rnn.predict(X_val, verbose=0)
            rnn_scores.append(-float(np.mean((pred.flatten() - y_val.flatten()) ** 2)))

        gs_lstm_rows.append({"param_optimiser": optimiser, "param_neurons": neurons,
                              "mean_test_score": np.mean(lstm_scores)})
        gs_rnn_rows.append({"param_optimiser": optimiser,  "param_neurons": neurons,
                             "mean_test_score": np.mean(rnn_scores)})

gs_lstm_df = pd.DataFrame(gs_lstm_rows)
gs_rnn_df  = pd.DataFrame(gs_rnn_rows)
gs_lstm_df.to_csv(os.path.join(SAVE_DIR, "gridsearch_lstm.csv"), index=False)
gs_rnn_df.to_csv(os.path.join(SAVE_DIR,  "gridsearch_rnn.csv"),  index=False)
print("    Saved gridsearch_lstm.csv  and  gridsearch_rnn.csv")

best_lstm_row    = gs_lstm_df.loc[gs_lstm_df["mean_test_score"].idxmax()]
best_rnn_row     = gs_rnn_df.loc[gs_rnn_df["mean_test_score"].idxmax()]

print(f"\n    LSTM Best params {{'neurons': {int(best_lstm_row['param_neurons'])}, "
      f"'optimiser': '{best_lstm_row['param_optimiser']}'}} "
      f"with Mean Square Root Error {best_lstm_row['mean_test_score']:.6e}")
print(f"    RNN  Best params {{'neurons': {int(best_rnn_row['param_neurons'])}, "
      f"'optimiser': '{best_rnn_row['param_optimiser']}'}} "
      f"with Mean Square Root Error {best_rnn_row['mean_test_score']:.6e}")

# Save best models from grid search
best_lstm_params = {"optimiser": best_lstm_row["param_optimiser"],
                    "neurons":   int(best_lstm_row["param_neurons"])}
best_rnn_params  = {"optimiser": best_rnn_row["param_optimiser"],
                    "neurons":   int(best_rnn_row["param_neurons"])}

best_lstm = build_lstm(optimiser=best_lstm_params["optimiser"],
                        neurons=best_lstm_params["neurons"],
                        n_inputs=15, n_features=N_FEATURES)
best_lstm.fit(X_train_gs, y_train_gs, epochs=100, batch_size=100,
              validation_split=0.2, callbacks=[es(15)], verbose=0)
best_lstm.save(os.path.join(SAVE_DIR, "lstm_best_grid.keras"))

best_rnn = build_rnn_multi(optimiser=best_rnn_params["optimiser"],
                             neurons=best_rnn_params["neurons"],
                             n_inputs=15, n_features=N_FEATURES)
best_rnn.fit(X_train_gs, y_train_gs, epochs=100, batch_size=100,
             validation_split=0.2, callbacks=[es(5)], verbose=0)
best_rnn.save(os.path.join(SAVE_DIR, "rnn_best_grid.keras"))
print("    Saved  lstm_best_grid.keras  and  rnn_best_grid.keras")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Keras Tuner — Hyperband on single-step LSTM
#     Searches units 32–448 (step 32), uses the best n_inputs found
#     The notebook searched n_inputs 10–30; we use a reduced set for runtime.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🔍  [5/7]  Keras Tuner Hyperband — single-step LSTM …")
print("    Searching input sizes [10, 15, 20, 24, 25, 30] …")

tuner_results = []
stop_early    = EarlyStopping(monitor="val_loss", patience=5, verbose=0)

input_size_candidates = [10, 15, 20, 24, 25, 30]

for n_inp in input_size_candidates:
    print(f"\n    → n_input = {n_inp}")

    # Rebuild model_builder closure for this n_inp
    def make_builder(n):
        def _builder(hp):
            from tensorflow.keras.layers import LSTM, Dense
            model = tf.keras.Sequential()
            model.add(LSTM(
                units=hp.Int("units", min_value=32, max_value=448, step=32),
                return_sequences=False,
                input_shape=(n, N_FEATURES)
            ))
            model.add(Dense(units=1))
            model.compile(optimizer="sgd", loss=tf.keras.losses.logcosh, metrics=["mse"])
            return model
        return _builder

    tuner = kt.Hyperband(
        make_builder(n_inp),
        objective="val_mse",
        max_epochs=20,
        directory="lstm_tuner_models",
        project_name=f"lstm_tuner_n{n_inp}",
        overwrite=True
    )

    X_tr_t, y_tr_t = preprocess_lstm(
        train_data, n_inputs=n_inp, n_predictions=1, n_features=N_FEATURES)
    tuner.search(X_tr_t, y_tr_t, epochs=n_inp, batch_size=100,
                 validation_split=0.2, callbacks=[stop_early], verbose=0)

    best_hps_t = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_m_t   = tuner.hypermodel.build(best_hps_t)
    hist_t     = best_m_t.fit(X_tr_t, y_tr_t, epochs=50, batch_size=50,
                               validation_split=0.3, callbacks=[stop_early],
                               verbose=0)
    val_loss = min(hist_t.history["val_loss"])

    print(f"    units={best_hps_t.get('units')},  val_loss={val_loss:.6e}")
    tuner_results.append({
        "inputs": n_inp,
        "units":  best_hps_t.get("units"),
        "val_loss": val_loss
    })

tuner_df = pd.DataFrame(tuner_results).sort_values("val_loss")
tuner_df.to_csv(os.path.join(SAVE_DIR, "tuner_single_step_results.csv"), index=False)
print("\n    Saved  tuner_single_step_results.csv")
print(tuner_df.to_string(index=False))

best_row     = tuner_df.iloc[0]
best_n_inp   = int(best_row["inputs"])
best_n_units = int(best_row["units"])
print(f"\n    ✅  Best single-step config: n_inputs={best_n_inp}, units={best_n_units}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Keras Tuner — Hyperband for 2-day prediction (7-step look-back)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🔍  [6/7]  Keras Tuner Hyperband — 2-day prediction LSTM …")
print("    units 192–252 (step 32), dense 10–50 (step 10) …")

X_train_2d, y_train_2d = preprocess_lstm(
    train_data, n_inputs=7, n_predictions=2, n_features=N_FEATURES)

tuner_2d_lstm = kt.Hyperband(
    model_builder_2day,
    objective="val_mse",
    max_epochs=20,
    directory="lstm_2d_tuner_models",
    project_name="lstm_tuner",
    overwrite=True
)

early_stop_2d = EarlyStopping(monitor="val_loss", patience=5,
                               verbose=0, mode="auto")

tuner_2d_lstm.search(
    X_train_2d, y_train_2d,
    epochs=20, batch_size=100,
    validation_split=0.2,
    callbacks=[early_stop_2d],
    verbose=0
)

best_hps_2d = tuner_2d_lstm.get_best_hyperparameters(num_trials=1)[0]
print(f"\n    The hyperparameter search is complete.")
print(f"    Optimal units in LSTM layers: {best_hps_2d.get('units')}")
print(f"    Optimal units in dense layer: {best_hps_2d.get('dense')}")

lstm_2d = tuner_2d_lstm.hypermodel.build(best_hps_2d)
lstm_2d.summary()

early_stop_fit = EarlyStopping(monitor="val_loss", patience=10,
                                verbose=1, mode="auto")
history_2d = lstm_2d.fit(
    X_train_2d, y_train_2d,
    epochs=60, batch_size=100,
    validation_split=0.2,
    callbacks=[early_stop_fit],
    verbose=0
)
lstm_2d.save(os.path.join(SAVE_DIR, "deep_lstm_2day.keras"))
print("    Saved  deep_lstm_2day.keras")

pd.DataFrame(history_2d.history).to_csv(
    os.path.join(SAVE_DIR, "history_deep_lstm_2day.csv"), index=False)

# Also save the best hyperparameters as JSON for the app to display
import json
best_hps_dict = {
    "units": int(best_hps_2d.get("units")),
    "dense": int(best_hps_2d.get("dense"))
}
with open(os.path.join(SAVE_DIR, "best_hps_2day.json"), "w") as f:
    json.dump(best_hps_dict, f)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  1-D CNN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🧠  [7/7]  1-D CNN (n_inputs=10, n_features=6) …")

X_train_cnn, y_train_cnn = preprocess_lstm(
    train_data, n_inputs=10, n_predictions=1, n_features=N_FEATURES)

model_cnn = build_cnn(n_inputs=10, n_features=N_FEATURES)
model_cnn.summary()

early_stop_cnn = EarlyStopping(monitor="val_loss", patience=10,
                                verbose=1, mode="auto")
history_cnn = model_cnn.fit(
    X_train_cnn, y_train_cnn,
    epochs=20, batch_size=100,
    validation_split=0.2,
    callbacks=[early_stop_cnn],
    verbose=0
)
model_cnn.save(os.path.join(SAVE_DIR, "cnn.keras"))
print("    Saved  cnn.keras")

pd.DataFrame(history_cnn.history).to_csv(
    os.path.join(SAVE_DIR, "history_cnn.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Final test MSE summary (matches notebook output block)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊  Test MSE Summary (normalised-space, 10-step sequences)")

X_te10, y_te10 = preprocess_lstm(
    test_data, n_inputs=10, n_predictions=1, n_features=N_FEATURES)

final_results = {}
for label, model in [
    ("LSTM (exploratory)",  model_lstm),
    ("RNN single feature",  model_rnn_single),
    ("CNN",                 model_cnn),
    ("LSTM best grid",      best_lstm),
]:
    try:
        # RNN single uses 1-feature sequences
        if label == "RNN single feature":
            _X, _y = preprocess_lstm(pd.DataFrame(test_data["Close"]),
                                      n_inputs=10, n_predictions=1, n_features=1)
        else:
            _X, _y = X_te10, y_te10
        pred = model.predict(_X, verbose=0)
        mse_val = float(np.mean((pred.flatten() - _y.flatten()) ** 2))
        final_results[label] = round(mse_val, 8)
        print(f"    {label:<30s}  MSE = {mse_val:.6e}")
    except Exception as e:
        print(f"    {label:<30s}  ⚠️  {e}")

pd.DataFrame(final_results, index=["MSE"]).to_csv(
    os.path.join(SAVE_DIR, "test_mse_summary.csv"))

print("\n✅  All models trained, tuned, and saved to ./saved_models/")
print("    Launch the app with:  streamlit run app.py")
