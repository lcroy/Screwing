import tensorflow as tf
import keras
import json
import os

from keras import layers
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import sys
sys.path.append("..")
from configure import Config
from utils import *


# build model
def conv1D_scr(input_shape, num_class):
    model = keras.Sequential()

    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    return model


if __name__ == "__main__":
    cfg = Config()

    # setup parameter based on the data source
    data_source = cfg.data_source
    if data_source =="raw_aursad_D":
        X_train, X_test, y_train, y_test = load_raw_data(cfg.raw_aursad_D_path, expand_flag=True)
        model_path = os.path.join(cfg.model_Conv1D_path, 'raw_model.h5')
        loss_img = os.path.join(cfg.Conv1D_loss_acc_fig_path, 'raw_loss.png')
        acc_img = os.path.join(cfg.Conv1D_loss_acc_fig_path, 'raw_acc.png')
        precision = "raw_Conv1D_precision"
        recall = "raw_Conv1D_recall"
        f1 = "raw_Conv1D_f1"
    else:
        X_train, X_test, y_train, y_test = load_feature_data(cfg.raw_aursad_path, expand_flag=True)
        model_path = os.path.join(cfg.model_Conv1D_path, 'feature_model.h5')
        loss_img = os.path.join(cfg.Conv1D_loss_acc_fig_path, 'feature_loss.png')
        acc_img = os.path.join(cfg.Conv1D_loss_acc_fig_path, 'feature_acc.png')
        precision = "feature_Conv1D_precision"
        recall = "feature_Conv1D_recall"
        f1 = "feature_Conv1D_f1"

    # construct conv1D
    model = conv1D_scr(X_train.shape[1:], cfg.num_class)
    #
    # compile model
    opt = tf.optimizers.Adam(cfg.lr)
    model.compile(optimizer=opt, loss=cfg.loss, metrics='acc')

    # the training will stop if the accuracy is not improved after "patinence" epochs - using early stopping for efficient
    callbacks = [keras.callbacks.EarlyStopping(patience=cfg.patience, restore_best_weights=True)]

    # training model
    history = model.fit(X_train, y_train, epochs=cfg.epochs, batch_size=cfg.batch_size, validation_data=(X_test,y_test), callbacks=callbacks)

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
