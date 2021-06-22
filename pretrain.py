import tensorflow.keras as keras
from tqdm import tqdm 
from sklearn.model_selection import KFold

def pretrain_finetune_evaluate(ds_load_result, pretrain_model, bs=4, epochs=40, verbose=0, kf_splits=5, init_uncertainty_times=5):
    train_ds, test_ds, preprocessing_layer, label_encoder = ds_load_result

    x_train, y_train = train_ds
    x_test, y_test = test_ds

    input_shape = x_train.shape[1:]
    output_len = len(label_encoder.classes_)

    x_train = preprocessing_layer(x_train)
    x_test = preprocessing_layer(x_test)

    accuracies = []
    kf = KFold(n_splits=kf_splits)
    for train_idx, val_idx in tqdm(kf.split(x_train)):
        for _ in range(init_uncertainty_times):
            _x_train, _x_val = x_train[train_idx], x_train[val_idx]
            _y_train, _y_val = y_train[train_idx], y_train[val_idx]

            # pretrain
            input_layer = keras.Input(shape=input_shape)

            x = pretrain_model(input_layer)

            output_layer = x

            model = keras.Model(input_layer, output_layer)

            model.compile(loss='mae', optimizer='adam')
            callbacks = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='loss')]

            model.fit(x_train, x_train, epochs=epochs, batch_size=2*bs, callbacks=callbacks, verbose=verbose)

            # fine tune
            base = keras.Model(model.inputs[0], model.get_layer('trunk').output)
            output_layer = keras.layers.Dense(output_len, activation='softmax')(base.output)

            model = keras.Model(base.inputs[0], output_layer)
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

            callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
            model.fit(_x_train, _y_train, validation_data=(_x_val, _y_val), epochs=epochs, batch_size=bs, validation_split=0.2, callbacks=callbacks, verbose=verbose)

            _, eval_acc = model.evaluate(x_test, y_test, verbose=verbose)
            accuracies += [eval_acc]

    return accuracies
