"""
models.py
=========
All four model architectures reproduced exactly from the notebook.

1. build_lstm()       — single LSTM layer, best params from GridSearchCV
                        (neurons=128, optimiser='adam', loss='logcosh')
2. build_rnn_single() — SimpleRNN on Close price only (1 feature)
3. build_rnn_multi()  — SimpleRNN on all 6 features
4. build_cnn()        — 1-D CNN (Conv1D → MaxPool → Conv1D → Dense → Dropout → Dense)
5. build_deep_lstm()  — two stacked LSTM layers for 2-day prediction,
                        architecture from Keras Tuner best_hps
                        (units=192, dense=40)

Hyperparameter search wrappers (KerasRegressor) are defined in tune.py so
they are available to GridSearchCV without importing Keras into the training
script unnecessarily.
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, SimpleRNN,
    Conv1D, MaxPool1D, Flatten
)


# ── 1. Single-layer LSTM (1-step prediction, 6 features) ─────────────────────

def build_lstm(optimiser: str = "adam",
               neurons: int   = 128,
               n_inputs: int  = 15,
               n_features: int = 6) -> Sequential:
    """
    Replicates the notebook's create_lstm_model() used inside GridSearchCV:

        def create_lstm_model(optimiser, neurons):
            model = Sequential()
            model.add(LSTM(units=neurons,
                           return_sequences=False,
                           input_shape=(15, 6)))
            model.add(Dense(units=1))
            model.compile(optimizer=optimiser,
                           loss='logcosh',
                           metrics='mse')
            return model

    n_inputs defaults to 15 to match the grid-search data preparation
    (X_train, y_train = preprocess_lstm(train_data, n_inputs=15, ...)).
    Pass n_inputs=10 when building the exploratory single-step model.
    """
    model = Sequential(name="LSTM")
    model.add(LSTM(units=neurons,
                   return_sequences=False,
                   input_shape=(n_inputs, n_features)))
    model.add(Dense(units=1))
    model.compile(optimizer=optimiser,
                  loss=tf.keras.losses.logcosh,
                  metrics=["mse"])
    return model


# ── 2. SimpleRNN — single feature (Close only) ───────────────────────────────

def build_rnn_single(optimiser: str  = "adam",
                     neurons: int    = 64,
                     n_inputs: int   = 10,
                     n_features: int = 1) -> Sequential:
    """
    Replicates model_rnn_single from the notebook:

        model_rnn_single = Sequential()
        model_rnn_single.add(SimpleRNN(units=64,
                                       return_sequences=False,
                                       input_shape=(10, 1)))
        model_rnn_single.add(Dense(units=1))
        model_rnn_single.compile(optimizer='adam',
                                  loss='logcosh',
                                  metrics='mse')
    """
    model = Sequential(name="RNN_single_feature")
    model.add(SimpleRNN(units=neurons,
                        return_sequences=False,
                        input_shape=(n_inputs, n_features)))
    model.add(Dense(units=1))
    model.compile(optimizer=optimiser,
                  loss=tf.keras.losses.logcosh,
                  metrics=["mse"])
    return model


# ── 3. SimpleRNN — multi-feature (n_features=5 in notebook section B) ────────

def build_rnn_multi(optimiser: str  = "adam",
                    neurons: int    = 64,
                    n_inputs: int   = 10,
                    n_features: int = 5) -> Sequential:
    """
    Replicates model_rnn_multi from the notebook (n_features=5 there):

        n_features = 5
        model_rnn_multi = Sequential()
        model_rnn_multi.add(SimpleRNN(units=64,
                                      return_sequences=False,
                                      input_shape=(10, n_features)))
        model_rnn_multi.add(Dense(units=1))
        model_rnn_multi.compile(optimizer='adam',
                                 loss='logcosh',
                                 metrics='mse')

    Also serves as create_rnn_model() for GridSearchCV (input_shape=(15, 6)):

        def create_rnn_model(optimiser, neurons):
            model_rnn_single = Sequential()
            model_rnn_single.add(SimpleRNN(units=neurons,
                                           return_sequences=False,
                                           input_shape=(15, 6)))
            model_rnn_single.add(Dense(units=1))
            model_rnn_single.compile(optimizer=optimiser,
                                      loss='logcosh',
                                      metrics='mse')
            return model_rnn_single
    """
    model = Sequential(name="RNN_multi_feature")
    model.add(SimpleRNN(units=neurons,
                        return_sequences=False,
                        input_shape=(n_inputs, n_features)))
    model.add(Dense(units=1))
    model.compile(optimizer=optimiser,
                  loss=tf.keras.losses.logcosh,
                  metrics=["mse"])
    return model


# ── 4. 1-D CNN ────────────────────────────────────────────────────────────────

def build_cnn(n_inputs: int   = 10,
              n_features: int = 6) -> Sequential:
    """
    Replicates model_cnn from the notebook exactly:

        model_cnn = Sequential()
        model_cnn.add(keras.layers.Conv1D(filters=64, kernel_size=3,
                                          activation='relu',
                                          input_shape=(10, 6)))
        model_cnn.add(keras.layers.MaxPool1D(pool_size=2))
        model_cnn.add(keras.layers.Conv1D(filters=32, kernel_size=3,
                                          activation='relu'))
        model_cnn.add(keras.layers.Flatten())
        model_cnn.add(Dense(units=128, activation='relu'))
        model_cnn.add(Dropout(rate=0.15))
        model_cnn.add(Dense(units=64, activation='relu'))
        model_cnn.add(Dense(units=1))
        model_cnn.compile(optimizer='adam', loss='mse', metrics=['mse'])

    Note: both Conv1D layers use kernel_size=3 (as in the notebook).
    """
    model = Sequential(name="CNN")
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu",
                     input_shape=(n_inputs, n_features)))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(rate=0.15))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


# ── 5. Deep LSTM for 2-day prediction ────────────────────────────────────────

def build_deep_lstm(units: int       = 192,
                    dense_units: int = 40,
                    n_inputs: int    = 7,
                    n_features: int  = 6,
                    n_outputs: int   = 2) -> Sequential:
    """
    Replicates the tuner-selected architecture from the notebook:

        lstm_2d = tuner_2d_lstm.hypermodel.build(best_hps)
        # best_hps: units=192, dense=40

    Which corresponds to create_2day_model with those hyperparameters:

        model.add(LSTM(units=192, return_sequences=True,  input_shape=(7, 6)))
        model.add(LSTM(units=192, return_sequences=False, input_shape=(7, 6)))
        model.add(Dense(units=40, activation='linear'))
        model.add(Dense(units=40, activation='linear'))
        model.add(Dense(units=2))
        model.compile(optimizer='adam', loss='logcosh', metrics='mse')
    """
    model = Sequential(name="DeepLSTM_2day")
    model.add(LSTM(units=units,
                   return_sequences=True,
                   input_shape=(n_inputs, n_features)))
    model.add(LSTM(units=units,
                   return_sequences=False))
    model.add(Dense(units=dense_units, activation="linear"))
    model.add(Dense(units=dense_units, activation="linear"))
    model.add(Dense(units=n_outputs))
    model.compile(optimizer="adam", loss=tf.keras.losses.logcosh, metrics=["mse"])
    return model


# ── Keras Tuner model builders ────────────────────────────────────────────────

def model_builder_single_step(hp):
    """
    Replicates model_builder() used with kt.Hyperband in the notebook:

        def model_builder(hp):
            model = Sequential()
            model.add(LSTM(units=hp.Int('units', min_value=32,
                                        max_value=448, step=32),
                           return_sequences=False,
                           input_shape=(n_input, 6)))
            model.add(Dense(units=1))
            model.compile(optimizer='sgd',
                           loss='logcosh',
                           metrics='mse')
            return model

    n_input is passed in as a global in the notebook; here we default to 10.
    The caller sets n_input on this function's closure before passing to tuner.
    """
    model = Sequential()
    model.add(LSTM(
        units=hp.Int("units", min_value=32, max_value=448, step=32),
        return_sequences=False,
        input_shape=(10, 6)
    ))
    model.add(Dense(units=1))
    model.compile(optimizer="sgd", loss=tf.keras.losses.logcosh, metrics=["mse"])
    return model


def model_builder_2day(hp):
    """
    Replicates create_2day_model() used with kt.Hyperband for 2-day prediction:

        def create_2day_model(hp):
            model = Sequential()
            model.add(LSTM(units=hp.Int('units', min_value=192,
                                        max_value=252, step=32),
                           return_sequences=True, input_shape=(7, 6)))
            model.add(LSTM(units=hp.Int('units', min_value=192,
                                        max_value=252, step=32),
                           return_sequences=False, input_shape=(7, 6)))
            model.add(Dense(units=hp.Int('dense', min_value=10,
                                         max_value=50, step=10),
                            activation='linear'))
            model.add(Dense(units=hp.Int('dense', min_value=10,
                                         max_value=50, step=10),
                            activation='linear'))
            model.add(Dense(units=2))
            model.compile(optimizer='adam', loss='logcosh', metrics='mse')
            return model
    """
    model = Sequential()
    model.add(LSTM(
        units=hp.Int("units", min_value=192, max_value=252, step=32),
        return_sequences=True,
        input_shape=(7, 6)
    ))
    model.add(LSTM(
        units=hp.Int("units", min_value=192, max_value=252, step=32),
        return_sequences=False
    ))
    model.add(Dense(
        units=hp.Int("dense", min_value=10, max_value=50, step=10),
        activation="linear"
    ))
    model.add(Dense(
        units=hp.Int("dense", min_value=10, max_value=50, step=10),
        activation="linear"
    ))
    model.add(Dense(units=2))
    model.compile(optimizer="adam", loss=tf.keras.losses.logcosh, metrics=["mse"])
    return model
