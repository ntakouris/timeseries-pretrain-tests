from functools import partial
import lstms, transformers

def lstm_models(lstm_units=[64, 32, 16], lstm_layers=[1,2], mlps=[[64, 32], [32, 16], [16, 8]], dropout=0.2):
    for lstm_layer in lstm_layers:
        for lstm_unit in lstm_units:
            for mlp in mlps:
                train_model = partial(lstms.classifier, lstm_units=lstm_unit, lstm_layers=lstm_layer, mlp=mlp, dropout=dropout)
                pretrain_model = partial(lstms.autoencoder, lstm_units=lstm_unit, lstm_layers=lstm_layer, mlps=len(mlp), dropout=dropout)

                yield train_model, pretrain_model, False

def transformer_models(heads=[1, 2, 4], head_sizes=[2, 4, 8], layers=[1, 2, 4], mlps=[[64, 32], [32, 16], [16, 8]], dropout=0.2):
    for num_heads in heads:
        for head_size in head_sizes:
            for num_layers in layers:
                for mlp in mlps:
                    train_model = partial(transformers.classifier, num_heads=num_heads, head_size=head_size, num_layers=num_layers, mlp=mlp, dropout=0.2)
                    pretrain_model = partial(transformers.autoencoder, num_heads=num_heads, head_size=head_size, num_layers=num_layers, mlps=len(mlp), dropout=0.2)

                    yield train_model, pretrain_model, True