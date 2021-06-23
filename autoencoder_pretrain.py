import tensorflow.keras as keras

def train_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, n_classes, model, bs=4, epochs=40, verbose=0, optimizer='adam', pool_trunk=False):
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

    x = base.output
    if pool_trunk:
        x = keras.layers.GlobalAveragePooling1D()(x)
    
    output_layer = keras.layers.Dense(
        n_classes, activation='softmax')(x)

    model = keras.Model(base.inputs[0], output_layer)
    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

    callbacks = [keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True)]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                batch_size=bs, callbacks=callbacks, verbose=verbose)

    eval_acc = model.evaluate(x_test, y_test, verbose=verbose, return_dict=True)['sparse_categorical_accuracy']
    return eval_acc
