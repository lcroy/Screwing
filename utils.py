import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# load dataset
def load_feature_data(dataset, expand_flag):
    # read the raw data file
    with open(dataset, "rb") as raw_file:
        raw_data = pickle.load(raw_file)

    # split data
    X_train, X_test, y_train, y_test = raw_data['X_train'], raw_data['X_test'],raw_data['y_train'], raw_data['y_test']

    # expand to 3-dim
    if expand_flag == True:
        X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

    # change label to one-hot
    # Y_train, Y_test = pd.get_dummies(Y_train), pd.get_dummies(Y_test)

    return X_train, X_test, y_train, y_test

# load dataset
def load_raw_data(dataset, expand_flag):
    # read the raw data file
    with open(dataset, "rb") as raw_file:
        raw_data = pickle.load(raw_file)

    # split data
    X, y = raw_data['X_t'], raw_data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # expand to 3-dim
    if expand_flag == True:
        X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

    # change label to one-hot
    # Y_train, Y_test = pd.get_dummies(Y_train), pd.get_dummies(Y_test)

    return X_train, X_test, y_train, y_test

# plot the accuracy and loss
def plot_loss_acc(history, loss_img, acc_img):
    # plot accuracy figure
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(acc_img)
    plt.show()

    # plot loss figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(loss_img)
    plt.show()
