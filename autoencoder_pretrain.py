import tensorflow.keras as keras

def lstm_autoencoder(input_x, lstm_units=8, lstm_layers=1, mlps=1, act='swish'):
    x = input_x
    for _ in range(lstm_layers-1):
        x = keras.layers.LSTM(lstm_units, return_sequences=True)(x)

    x = keras.layers.LSTM(lstm_units, name='trunk')(input_x)

    x = keras.layers.RepeatVector(input_x.shape[1])(x)

    for _ in range(lstm_layers):
        x = keras.layers.LSTM(lstm_units, return_sequences=True)(x)

    _mlp_dim = input_x.shape[-1]
    for _ in range(mlps-1):
        x = keras.layers.TimeDistributed(keras.layers.Dense(_mlp_dim, activation=act))(x)

    return keras.layers.TimeDistributed(keras.layers.Dense(input_x.shape[-1]))(x)


def train_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, n_classes, model, bs=4, epochs=40, verbose=0, optimizer='adam'):
    input_shape = x_train.shape[1:]

    # pretrain
    input_layer = keras.Input(shape=input_shape)

    x = model(input_layer)

    output_layer = x

    model = keras.Model(input_layer, output_layer)

    model.compile(loss='mse', optimizer=optimizer)
    callbacks = [keras.callbacks.EarlyStopping(
        patience=3, restore_best_weights=True, monitor='loss')]

    model.fit(x_train, x_train, epochs=epochs,
                batch_size=bs, callbacks=callbacks, verbose=verbose)

    # fine tune
    base = keras.Model(
        model.inputs[0], model.get_layer('trunk').output)
    output_layer = keras.layers.Dense(
        n_classes, activation='softmax')(base.output)

    model = keras.Model(base.inputs[0], output_layer)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

    callbacks = [keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True)]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                batch_size=bs, callbacks=callbacks, verbose=verbose)

    eval_acc = model.evaluate(x_test, y_test, verbose=verbose, return_dict=True)['sparse_categorical_accuracy']
    return eval_acc
