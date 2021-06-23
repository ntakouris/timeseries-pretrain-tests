import tensorflow.keras as keras
from tqdm import tqdm
from sklearn.model_selection import KFold


def pretrain_finetune_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, n_classes, pretrain_model, bs=4, epochs=40, verbose=0, optimizer='adam'):
    input_shape = x_train.shape[1:]

    # pretrain
    input_layer = keras.Input(shape=input_shape)

    x = pretrain_model(input_layer)

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
