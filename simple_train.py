import tensorflow.keras as keras

def lstm_classifier(x, lstm_units=8, lstm_layers=1, mlp=[], act='swish'):
    for _ in range(lstm_layers-1):
        x = keras.layers.LSTM(lstm_units, return_sequences=True)(x)

    x = keras.layers.LSTM(lstm_units)(x)

    for d in mlp:
        x = keras.layers.Dense(d, activation=act)(x)

    return x

def train_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, n_classes, model, bs=4, epochs=40, verbose=0, optimizer='adam'):
    input_shape = x_train.shape[1:]

    input_layer = keras.Input(shape=input_shape)
    
    x = model(input_layer)

    output_layer = keras.layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(input_layer, output_layer)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

    callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=bs, callbacks=callbacks, verbose=verbose)

    eval_acc = model.evaluate(x_test, y_test, verbose=verbose, return_dict=True)['sparse_categorical_accuracy']
    return eval_acc
