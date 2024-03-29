import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import utils

# load dataset
def load_feature_data(dataset, expand_flag):
    # read the raw data file
    with open(dataset, "rb") as raw_file:
        raw_data = pickle.load(raw_file)

    # split data
    X_train, X_test, y_train, y_test = raw_data['X_train'], raw_data['X_test'], raw_data['y_train'], raw_data['y_test']

    print(y_train.value_counts())
    print(y_test.value_counts())

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

    print(y_train.value_counts())
    print(y_test.value_counts())

    # expand to 3-dim
    if expand_flag == True:
        X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

    # change label to one-hot
    # Y_train, Y_test = pd.get_dummies(Y_train), pd.get_dummies(Y_test)

    return X_train, X_test, y_train, y_test


#===============================original data (process + task)==========================================
# load dataset
def load_org_data_only_process(dataset, expand_flag):
    # read the raw data file
    with open(dataset, "rb") as raw_file:
        raw_data = pickle.load(raw_file)

    # obtain torque and lable
    temp = pd.DataFrame([[item['torque'], item['label']] for item in raw_data], columns=["torque","label"])
    temp_X, temp_y = temp['torque'], temp['label']

    # format the torque and label
    X, y = [], []
    for item in temp_X:
        values = item.values
        X.append(values)
    X = pd.DataFrame(X).fillna(0)

    for item in temp_y:
        values = item[0]
        y.append(values)
    y = pd.DataFrame(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

    print(y_train.value_counts())
    print(y_test.value_counts())

    # expand to 3-dim
    if expand_flag == True:
        X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

    print(X_train.shape)

    return X_train, X_test, y_train, y_test


# plot the accuracy and loss
def plot_loss_acc(history, loss_img, acc_img):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(acc_img)
    plt.close()

    # plot loss figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(loss_img)
    plt.close()

count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0

def restructure_data(feature_data):
    global count_1, count_2,count_3,count_4,count_5
    feature = []
    for item in feature_data:
        # for au data
        # values = item.values
        # feature.append(values)

        # for ConvLSTM2D of aau data
        # if len(item) > 5020:
        #     item = item[:5020]

        feature.append(item)
    feature = pd.DataFrame(feature).fillna(0)

    return feature

def load_org_data_process_task(dataset, expand_flag):
    # read the raw data file
    with open(dataset, "rb") as raw_file:
        raw_data = pickle.load(raw_file)

    # obtain torque and lable
    # for flt or cln data
    # temp = pd.DataFrame([[item['torque'],
    #                       item['tcp_pose_0'], item['tcp_pose_1'], item['tcp_pose_2'], item['tcp_pose_3'], item['tcp_pose_4'], item['tcp_pose_5'],
    #                       item['tcp_speed_0'], item['tcp_speed_1'], item['tcp_speed_2'], item['tcp_speed_3'], item['tcp_speed_4'], item['tcp_speed_5'],
    #                       item['tcp_force_0'], item['tcp_force_1'], item['tcp_force_2'], item['tcp_force_3'], item['tcp_force_4'], item['tcp_force_5'],
    #                       item['label']] for item in raw_data], columns=["torque", "tcp_pose_0", "tcp_pose_1", "tcp_pose_2", "tcp_pose_3", "tcp_pose_4", "tcp_pose_5",
    #                                                                      "tcp_speed_0", "tcp_speed_1", "tcp_speed_2", "tcp_speed_3", "tcp_speed_4", "tcp_speed_5",
    #                                                                      "tcp_force_0", "tcp_force_1", "tcp_force_2", "tcp_force_3", "tcp_force_4", "tcp_force_5", "label"])


    # X, y = [], []
    # X.append(restructure_data(temp['torque']))
    # X.append(restructure_data(temp['tcp_pose_0']))
    # X.append(restructure_data(temp['tcp_pose_1']))
    # X.append(restructure_data(temp['tcp_pose_2']))
    # X.append(restructure_data(temp['tcp_pose_3']))
    # X.append(restructure_data(temp['tcp_pose_4']))
    # X.append(restructure_data(temp['tcp_pose_5']))
    # X.append(restructure_data(temp['tcp_speed_0']))
    # X.append(restructure_data(temp['tcp_speed_1']))
    # X.append(restructure_data(temp['tcp_speed_2']))
    # X.append(restructure_data(temp['tcp_speed_3']))
    # X.append(restructure_data(temp['tcp_speed_4']))
    # X.append(restructure_data(temp['tcp_speed_5']))
    # X.append(restructure_data(temp['tcp_force_0']))
    # X.append(restructure_data(temp['tcp_force_1']))
    # X.append(restructure_data(temp['tcp_force_2']))
    # X.append(restructure_data(temp['tcp_force_3']))
    # X.append(restructure_data(temp['tcp_force_4']))
    # X.append(restructure_data(temp['tcp_force_5']))


    # for aau data
    temp = pd.DataFrame([[item['torque'],
                          item['current'], item['angle'], item['depth'], item['label']] for item in raw_data], 
                          columns=["torque", "current", 'angle', 'depth', "label"])


    # print(temp)

    X, y = [], []
    X.append(restructure_data(temp['torque']))
    X.append(restructure_data(temp['current']))
    X.append(restructure_data(temp['angle']))
    X.append(restructure_data(temp['depth']))

    print(count_1,count_2,count_3,count_4,count_5)
    

    X = np.dstack(X)
    print(X.shape)

    for item in temp['label']:
        # values = item[0]
        y.append(item)
    y = pd.DataFrame(y)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

    print(y_train.value_counts())
    print(y_test.value_counts())

    # # expand to 3-dim
    # if expand_flag == True:
    #     X_train, X_test = np.expand_dims(X_train,-1), np.expand_dims(X_test,-1)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test

