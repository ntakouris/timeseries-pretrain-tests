import os
import sklearn
import numpy as np
import tensorflow.keras as keras

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_processing import from_nested_to_3d_numpy


def load_classification_dataset(ds_path=os.path.join(os.path.dirname(__file__), 'datasets'), ds_name='BasicMotions'):
    ds_name = 'BasicMotions'

    train_x_df, train_y_labels = load_from_tsfile_to_dataframe(
        os.path.join(ds_path, f'{ds_name}/{ds_name}_TRAIN.ts')
    )

    test_x_df, test_y_labels = load_from_tsfile_to_dataframe(
        os.path.join(ds_path, f'{ds_name}/{ds_name}_TEST.ts')
    )

    train_x_np = from_nested_to_3d_numpy(train_x_df)
    test_x_np = from_nested_to_3d_numpy(test_x_df)

    # convert to bs, seq len, feat
    train_x_np = np.transpose(train_x_np, (0, 2, 1))
    test_x_np = np.transpose(test_x_np, (0, 2, 1))

    preprocessing_layer = keras.layers.experimental.preprocessing.Normalization()
    preprocessing_layer.adapt(train_x_np)

    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(train_y_labels)

    train_y_np = label_encoder.transform(train_y_labels)
    test_y_np = label_encoder.transform(test_y_labels)

    train_ds = (train_x_np, train_y_np)
    test_ds = (test_x_np, test_y_np)

    return train_ds, test_ds, preprocessing_layer, label_encoder
