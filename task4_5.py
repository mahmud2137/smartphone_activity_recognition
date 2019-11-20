import numpy as np 
import pandas as pd
# import keras 
# import keras.backend as K
from tensorflow import keras

import sys
sys.modules['keras'] = keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM, Softmax
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sys

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


################################
#Loading and Preprocessing Data
###############################
train_dir = r'har_dataset/train/'
test_dir = r'har_dataset/test/'

def get_sensor_value_combined(train_test, acc_gyro):
    #this function will reshape data in the form (sample, timestamp, feature) for LSTM input
    path = train_dir if train_test=='train' else test_dir
    value_axes = {}
    for axis in ['x','y','z']:
        value_axes[axis] = pd.read_csv(path + r'Inertial Signals/'+ f'body_{acc_gyro}_x_{train_test}.txt',sep=r"\s+", header=None)

    value_axes = np.dstack((value_axes['x'].values , value_axes['y'].values, value_axes['z'].values))
    return value_axes

acc_train = get_sensor_value_combined('train', 'acc')
gyro_train = get_sensor_value_combined('train', 'gyro')
acc_test = get_sensor_value_combined('test', 'acc')
gyro_test = get_sensor_value_combined('test', 'gyro')


###################
#getting the labels
###################
train_labels = pd.read_csv(train_dir+'y_train.txt', names=['label'])
test_labels = pd.read_csv(test_dir+'y_test.txt', names=['label'])
train_labels = train_labels.label.values
test_labels = test_labels.label.values
train_labels = to_categorical(train_labels, num_classes = 7)[:,1:]
test_labels = to_categorical(test_labels, num_classes = 7)[:,1:]




def LSTM_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(100, input_shape = input_shape, return_sequences = True, activation = 'relu'))
    model.add(LSTM(100, return_sequences = True, activation = 'relu'))
    model.add(LSTM(100, return_sequences = False, activation = 'relu'))
    model.add(Dense(output_size))
    model.add(Softmax())

    return model

LSTM_gyr = LSTM_model(input_shape = gyro_train.shape[1:], output_size=6)
opt = keras.optimizers.Adam(lr=0.001 , beta_1=0.9, beta_2=0.999)
LSTM_gyr.compile(loss = 'categorical_crossentropy',
            optimizer = opt,
            metrics = ['accuracy'])

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1 )

batch_size = 256
epochs = 100
history = LSTM_gyr.fit(gyro_train, train_labels,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_split = 0.15,
                    callbacks = [stop_early],
                    shuffle = True,
                    verbose = 2)


scores = LSTM_gyr.evaluate(gyro_test, test_labels)
print('Test loss: %.2f' %scores[0])
print('Test accuracy: %0.2f' %scores[1])

def plot_train_and_val(history):
    plt.figure(figsize=(10,5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig('Model_accuracy.png')
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig('Model_loss.png')

plot_train_and_val(history)


    