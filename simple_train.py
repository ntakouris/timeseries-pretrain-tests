import tensorflow.keras as keras
from tqdm import tqdm 
from sklearn.model_selection import KFold

def train_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, n_classes, trunk, bs=4, epochs=40, verbose=0, optimizer='adam'):
    input_shape = x_train.shape[1:]

    input_layer = keras.Input(shape=input_shape)
    
    x = trunk(input_layer)

    output_layer = keras.layers.Dense(n_classes, activation='softmax')(x)

    model = keras.Model(input_layer, output_layer)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

    callbacks = [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=bs, callbacks=callbacks, verbose=verbose)

    eval_acc = model.evaluate(x_test, y_test, verbose=verbose, return_dict=True)['sparse_categorical_accuracy']
    return eval_acc
