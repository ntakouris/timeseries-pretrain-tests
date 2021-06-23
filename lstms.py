import tensorflow.keras as keras

def autoencoder(input_x, lstm_units=8, lstm_layers=1, mlps=1, act='swish', dropout=0.2):
    x = input_x
    for _ in range(lstm_layers-1):
        x = keras.layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(x)

    x = keras.layers.LSTM(lstm_units, name='trunk')(input_x)

    x = keras.layers.RepeatVector(input_x.shape[1])(x)

    for _ in range(lstm_layers):
        x = keras.layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(x)

    _mlp_dim = input_x.shape[-1]
    for _ in range(mlps-1):
        x = keras.layers.TimeDistributed(keras.layers.Dense(_mlp_dim, activation=act))(x)

    return keras.layers.TimeDistributed(keras.layers.Dense(input_x.shape[-1]))(x)


def classifier(x, lstm_units=8, lstm_layers=1, mlp=[], act='swish', dropout=0.2):
    for _ in range(lstm_layers-1):
        x = keras.layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(x)

    x = keras.layers.LSTM(lstm_units, dropout=dropout)(x)

    for d in mlp:
        x = keras.layers.Dense(d, activation=act)(x)

    return x