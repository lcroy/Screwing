import tensorflow as tf
import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras import layers
from configure import Config
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import json



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

def conv1D_scr(input_shape, num_class):
    model = keras.Sequential()
    model.add(layers.Conv1D(32, 7, input_shape=input_shape, activation='relu', padding='same'))
    model.add(layers.MaxPool1D(3))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv1D(32, 7, activation='relu', padding='same'))
    model.add(layers.MaxPool1D(3))
    model.add(layers.Dropout(0.5))

    model.add(layers.GlobalAvgPool1D())
    model.add(layers.Dense(num_class, activation='softmax'))

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
    plt.savefig(cfg.acc_fig_path)
    plt.show()

    # plot loss figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(cfg.loss_fig_path)
    plt.show()

if __name__ == "__main__":
    cfg = Config()

    # split data
    X_train, X_test, Y_train, Y_test = load_data(cfg.raw_aursad_path)

    # construct conv1D
    model = conv1D_scr(X_train.shape[1:], cfg.num_class)
    #
    # compile model
    opt = tf.optimizers.Adam(cfg.lr)
    model.compile(optimizer=opt, loss=cfg.loss, metrics='acc')

    # training model
    history = model.fit(X_train, Y_train, epochs=cfg.epochs, batch_size=cfg.batch_size, validation_data=(X_test,Y_test))

    # save model
    model.save(cfg.model_Conv1D_path)

    # plot the acc and loss
    plot_loss_acc(history, cfg)

    # get f1, precision and recall scores
    model = keras.models.load_model(cfg.model_Conv1D_path)

    y_pred1 = model.predict(X_test)
    y_pred = np.argmax(y_pred1, axis=1)

    # save f1, precision, and recall scores
    scores = {"precision": precision_score(Y_test, y_pred, average="macro"),
              "recall": recall_score(Y_test, y_pred, average="macro"),
              "f1": f1_score(Y_test, y_pred, average="macro")}
    with open(cfg.scores_file_path, 'w') as outfile:
        json.dump(scores, outfile)
    outfile.close()
