import tensorflow.keras as keras
import tensorflow.keras.backend as K

import numpy as np

def autoencoder(input_x, num_heads=2, head_size=4, num_layers=1, act='swish', mlps=1, dropout=0.2):
    x = TransformerEncoder(num_heads=num_heads, head_size=head_size, ff_dim=None, num_layers=num_layers, dropout=dropout, name='trunk')(input_x)

    _mlp_dim = input_x.shape[-1]
    for _ in range(mlps-1):
        x = keras.layers.TimeDistributed(keras.layers.Dense(_mlp_dim, activation=act))(x)

    return keras.layers.TimeDistributed(keras.layers.Dense(input_x.shape[-1]))(x)

def classifier(x, num_heads=2, head_size=4, num_layers=1, act='swish', mlp=[], dropout=0.2):
    x = TransformerEncoder(num_heads=num_heads, head_size=head_size, ff_dim=None, num_layers=num_layers, dropout=dropout)(x)
    head = keras.layers.GlobalAveragePooling1D()(x) 

    head = keras.layers.Dropout(dropout)(head)

    for dim in mlp:
        head = keras.layers.Dense(dim)(head)

    return head


class SinusoidalPositionalEncoding(keras.Model):
    def __init__(self, name='SinusoidalPositionalEncoding', **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(position, d_model):
        angle_rads = SinusoidalPositionalEncoding.get_angles(np.arange(position)[:, np.newaxis],
                                                             np.arange(d_model)[
            np.newaxis, :],
            d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return K.cast(pos_encoding, dtype='float32')

    def build(self, input_shape):
        self.position = input_shape[-2]
        self.d_model = input_shape[-1]

    def call(self, inputs, training, **kwargs):
        pos_enc = SinusoidalPositionalEncoding.positional_encoding(
            self.position, self.d_model)
        return inputs + pos_enc

class SelfAttention(keras.Model):
    def __init__(self, name='SelfAttention', num_heads=1, head_size=32, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=dropout, **kwargs)

    def call(self, inputs, **kwargs):
        x = inputs
        return self.attention(query=x, key=x, value=x, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape


class SelfAttentionBlock(keras.Model):
    def __init__(self, name='SelfAttentionBlock', num_heads=2, head_size=128, ff_dim=None, dropout=0, **kwargs):
        super().__init__(name=name, **kwargs)

        if ff_dim is None:
            ff_dim = head_size

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff_conv1 = keras.layers.Conv1D(
            filters=ff_dim, kernel_size=1, activation='relu')
        # self.ff_conv2 at build()
        self.ff_dropout = keras.layers.Dropout(dropout)
        self.ff_norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.ff_conv2 = keras.layers.Conv1D(
            filters=input_shape[-1], kernel_size=1)

    def call(self, inputs, training, **kwargs):
        x = self.attention(inputs, inputs, **kwargs)
        x = self.attention_dropout(x, training=training, **kwargs)
        x = self.attention_norm(inputs + x, **kwargs)

        x = self.ff_conv1(x, **kwargs)
        x = self.ff_conv2(x, **kwargs)
        x = self.ff_dropout(x, training=training, **kwargs)

        x = self.ff_norm(inputs + x, **kwargs)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class TransformerEncoder(keras.Model):
    def __init__(self, name='TransformerEncoder', num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0.1, **kwargs):
        super().__init__(name=name, **kwargs)
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.attention_layers = [SelfAttentionBlock(
            num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)]

    def call(self, inputs, training, **kwargs):
        x = inputs
        for attention_layer in self.attention_layers:
            x = attention_layer(x, training, **kwargs)

        return x