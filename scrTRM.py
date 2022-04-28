import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras import layers
from configure import Config
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import json
from tensorflow import keras
from tensorflow.keras import layers


# load dataset
def load_data(dataset):
    # read the raw data file
    with open(dataset, "rb") as raw_file:
        raw_data = pickle.load(raw_file)

    # split data
    X_train, X_test, Y_train, Y_test = raw_data['X_train'], raw_data['X_test'],raw_data['y_train'], raw_data['y_test']

    # expand to 3-dim
    X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

    # change label to one-hot
    # Y_train, Y_test = pd.get_dummies(Y_train), pd.get_dummies(Y_test)

    return X_train, X_test, Y_train, Y_test


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes=4,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    cfg = Config()
    # split data
    x_train, x_test, y_train, y_test = load_data(cfg.raw_aursad_path)

    input_shape = x_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
        n_classes=cfg.num_class
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )

    model.evaluate(x_test, y_test, verbose=1)


# # convert to embedding vector sequence
# def pre_seq(image, patch_size, weight):
#     patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1,-2)
#     patch_embedding = patch * weight
#     return patch_embedding
#
#
# if __name__ == "__main__":
#     cfg = Config()
#
#     # test pre_seq
#     bs, ic, image_h, image_w = 1, 1, 1, 469
#     patch_size = 1
#     model_dim = 512
#     patch_depth = patch_size * patch_size * ic
#     seq = torch.randn(bs, ic, image_h, image_w)
#     weight = torch.randn(patch_depth, model_dim)
#     max_num_token = 470
#     num_classes = 4
#
#     label = torch.randint(4, (bs,))
#
#     # embedding sequence
#     patch_embedding = pre_seq(seq, patch_size, weight)
#
#     # add CLS token embedding
#     cls_token_embedding = torch.randn(bs, 1, model_dim, requires_grad=True)
#     token_embedding = torch.cat([cls_token_embedding, patch_embedding], dim=1)
#
#     # add position embedding
#     position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
#     seq_len = token_embedding.shape[1]
#     position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
#     token_embedding += position_embedding
#
#     # pass embedding to transformer encoder
#     encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim,nhead=8)
#     transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#     encoder_output = transformer_encoder(token_embedding)
#
#     # do classification
#     cls_token_output = encoder_output[:, 0, :]
#     linear_layer = nn.Linear(model_dim, num_classes)
#     logits = linear_layer(cls_token_output)
#
#     loss_fn = nn.CrossEntropyLoss()
#     loss = loss_fn(logits, label)
#
#     print(loss)
#
#     # # split data
#     # X_train, X_test, Y_train, Y_test = load_data(cfg.raw_aursad_path)
#     #
#     # print(X_train.shape)
