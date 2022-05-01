import tensorflow as tf
import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json

from keras import layers
from configure import Config
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
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


# plot the accuracy and loss
def plot_loss_acc(history, cfg):
    # plot accuracy figure
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(cfg.TRM_acc_fig_path)
    plt.show()

    # plot loss figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(cfg.TRM_loss_fig_path)
    plt.show()


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

    # split data
    x_train, x_test, y_train, y_test = load_data(cfg.raw_aursad_path)

    # data sample shape
    input_shape = x_train.shape[1:]

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
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=cfg.epochs, batch_size=cfg.batch_size, callbacks=callbacks)

    # save model
    model.save(cfg.model_TRM_path)

    # plot the acc and loss
    plot_loss_acc(history, cfg)

    # get f1, precision and recall scores
    model = keras.models.load_model(cfg.model_TRM_path)

    y_pred1 = model.predict(x_test)
    y_pred = np.argmax(y_pred1, axis=1)

    # save f1, precision, and recall scores
    scores = {"TRM_precision": precision_score(y_test, y_pred, average="macro"),
              "TRM_recall": recall_score(y_test, y_pred, average="macro"),
              "TRM_f1": f1_score(y_test, y_pred, average="macro")}
    with open(cfg.scores_file_path, 'a') as outfile:
        json.dump(scores, outfile)
    outfile.close()