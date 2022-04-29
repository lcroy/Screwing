import tensorflow as tf
import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import pandas as pd

from keras import layers
from configure import Config
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


# load dataset
def load_data(dataset):
    # read the raw data file
    with open(dataset, "rb") as raw_file:
        raw_data = pickle.load(raw_file)

    # split data
    X_train, X_test, Y_train, Y_test = raw_data['X_train'], raw_data['X_test'],raw_data['y_train'], raw_data['y_test']

    # expand to 3-dim
    # X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

    # change label to one-hot
    # Y_train, Y_test = pd.get_dummies(Y_train), pd.get_dummies(Y_test)

    # print(len(X_train), len(X_test), len(Y_train), len(Y_test))
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    return X_train, X_test, Y_train, Y_test


def DNN_scr(input_shape, cfg):
    model = keras.Sequential()
    model.add(layers.Dense(units=cfg.units_h1, input_dim=input_shape, activation='relu'))
    model.add(layers.Dense(units=cfg.units_h2, activation='relu'))
    model.add(layers.Dense(units=cfg.units_h3, activation='relu'))
    model.add(layers.Dense(units=cfg.units_h4, activation='relu'))
    model.add(layers.Dense(cfg.num_class, activation='softmax'))

    print(model.summary())

    return model

# plot the accuracy and loss
def plot_loss_acc(history, cfg):
    # plot accuracy figure
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(cfg.DNN_loss_fig_path)
    plt.show()

    # plot loss figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(cfg.DNN_loss_fig_path)
    plt.show()


if __name__ == "__main__":
    cfg = Config()

    # split data
    X_train, X_test, Y_train, Y_test = load_data(cfg.raw_aursad_path)

    # construct DNN
    print(X_train.shape[1])
    model = DNN_scr(X_train.shape[1], cfg)
    #
    # compile model
    opt = tf.optimizers.Adam(cfg.lr)
    model.compile(optimizer=opt, loss=cfg.loss, metrics='acc')

    # training model
    history = model.fit(X_train, Y_train, epochs=cfg.epochs, batch_size=cfg.batch_size, validation_data=(X_test,Y_test))

    # save model
    model.save(cfg.model_DNN_path)

    # plot the acc and loss
    plot_loss_acc(history, cfg)

    # get f1, precision and recall scores
    model = keras.models.load_model(cfg.model_DNN_path)

    y_pred1 = model.predict(X_test)
    y_pred = np.argmax(y_pred1, axis=1)

    # save f1, precision, and recall scores
    scores = {"DNN_precision": precision_score(Y_test, y_pred, average="macro"),
              "DNN_recall": recall_score(Y_test, y_pred, average="macro"),
              "DNN_f1": f1_score(Y_test, y_pred, average="macro")}
    with open(cfg.scores_file_path, 'w') as outfile:
        json.dump(scores, outfile)
    outfile.close()

