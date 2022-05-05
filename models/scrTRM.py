import tensorflow as tf
import keras
import json
import os
import argparse

from keras import layers
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import sys
sys.path.append("..")
from configure import Config
from utils import *


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


def TRM_scr(
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
    # get configure file
    cfg = Config()

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--is_raw_data", default=True, type=bool, required=True,
                        help="Select raw data or feature selected data")
    args = parser.parse_args()

    # split data source
    if args.is_raw_data == True:
        X_train, X_test, y_train, y_test = load_raw_data(cfg.raw_aursad_path, expand_flag=False)
    else:
        X_train, X_test, y_train, y_test = load_feature_data(cfg.feature_aursad_path, expand_flag=False)

    # set up parameters
    model_path, loss_img, acc_img, precision, recall, f1 = cfg.model_parameters_set("TRM", args.is_raw_data)

    # data sample shape
    input_shape = X_train.shape[1:]

    # initial a Transformer model
    model = TRM_scr(
        input_shape,
        head_size=cfg.head_size,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        num_transformer_blocks=cfg.num_transformer_blocks,
        mlp_units=cfg.mlp_units,
        mlp_dropout=cfg.mlp_dropout,
        dropout=cfg.dropout,
        n_classes=cfg.num_class
    )

    # compile model
    opt = tf.optimizers.Adam(cfg.lr)
    model.compile(loss=cfg.loss, optimizer=opt, metrics='acc')

    # get model summary
    model.summary()

    # the training will stop if the accuracy is not improved after "patinence" epochs - using early stopping for efficient
    callbacks = [keras.callbacks.EarlyStopping(patience=cfg.patience, restore_best_weights=True)]

    # training model 
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=cfg.epochs, batch_size=cfg.batch_size, callbacks=callbacks)

    # save model
    model.save(model_path)

    # plot the acc and loss
    plot_loss_acc(history, loss_img, acc_img)

    # get f1, precision and recall scores
    model = keras.models.load_model(model_path)

    y_pred1 = model.predict(X_test)
    y_pred = np.argmax(y_pred1, axis=1)

    # save f1, precision, and recall scores
    scores = {precision: precision_score(y_test, y_pred, average="macro"),
              recall: recall_score(y_test, y_pred, average="macro"),
              f1: f1_score(y_test, y_pred, average="macro")}
    with open(cfg.scores_file_path, 'a') as outfile:
        json.dump(scores, outfile)
    outfile.close()