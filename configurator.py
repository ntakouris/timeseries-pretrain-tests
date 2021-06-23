import simple_train, autoencoder_pretrain
from functools import partial

def lstm_models(lstm_units=[64, 32, 16], lstm_layers=[1,2], mlps=[[64, 32], [32, 16], [16, 8]]):
    for lstm_layer in lstm_layers:
        for lstm_unit in lstm_units:
            for mlp in mlps:
                train_model = partial(simple_train.lstm_classifier, lstm_units=lstm_unit, lstm_layers=lstm_layer, mlp=mlp)
                pretrain_model = partial(autoencoder_pretrain.lstm_autoencoder, lstm_units=lstm_unit, lstm_layers=lstm_layer, mlps=len(mlp))

                yield train_model, pretrain_model
