import numpy as np 
import pandas as pd
# import keras 
# import keras.backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import sys
sys.modules['keras'] = keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM, Softmax, RepeatVector
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sys
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from Output_label_task_1 import *

def plot_train_and_val(train, val, metrics = 'accuracy', model_name = '_'):
    plt.figure(figsize=(10,5))
    plt.plot(train)
    plt.plot(val)
    plt.title(f'{model_name}_{metrics}')
    plt.ylabel(metrics)
    plt.xlabel('epochs')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.show()
    plt.savefig(f'{model_name}_{metrics}.jpg')


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

#getting the categorical label or Label 1
train_labels_cat = map_label_1(train_labels, numeric_label=True)  #catagorigcal label for train
test_labels_cat = map_label_1(test_labels, numeric_label=True)
train_labels_cat = to_categorical(train_labels_cat, num_classes = 4)
test_labels_cat = to_categorical(test_labels_cat, num_classes = 4)

#getting the mobility score for regression label
train_mobility_score = map_mobility_score(train_labels)
test_mobility_score = map_mobility_score(test_labels)


#########################################
#Building the 2 input and 2 output model
############################################

input1 = Input(shape = acc_train.shape[1:], name = 'acc_input')
input2 = Input(shape = gyro_train.shape[1:], name = 'gyr_input')

lstm1_acc = LSTM(100, return_sequences = False, activation='relu')(input1)
lstm1_gyr = LSTM(100, return_sequences = False, activation='relu')(input2)
merged = concatenate([lstm1_acc, lstm1_gyr])
merged_rep = RepeatVector(acc_train.shape[1])(merged)
lstm2_m = LSTM(100, return_sequences = True, activation='relu')(merged_rep)
lstm3_m = LSTM(100, return_sequences = True, activation='relu')(lstm2_m)
lstm4_op1 = LSTM(100, return_sequences = False, activation='relu')(lstm3_m)
lstm4_op2 = LSTM(100, return_sequences = False, activation='relu')(lstm3_m)
op1 = Dense(4, activation = 'relu')(lstm4_op1)
op2 = Dense(1, activation = 'relu', name = 'mobility_score')(lstm4_op2)
op1 = Softmax(name = 'activiy')(op1)


model = Model(inputs = [input1, input2], outputs = [op1, op2])
model.summary()
# plot_model(model, to_file='model.png')


###################################
#Training and Evaluating the model
#####################################


# opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss = ['categorical_crossentropy','mse'],
            optimizer = 'RMSprop',
            metrics = ['accuracy'])

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1 )

batch_size = 256
epochs = 50
history = model.fit([acc_train,gyro_train], [train_labels_cat,train_mobility_score],
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_split = 0.15,
                    # callbacks = [stop_early],
                    shuffle = True,
                    verbose = 2)



scores = model.evaluate([acc_test, gyro_test], [test_labels_cat, test_mobility_score])
scores
for i,metric in enumerate(model.metrics_names):
    print(metric, ': ', scores[i])

plot_train_and_val(train = history.history['activiy_accuracy'],
                    val = history.history['val_activiy_accuracy'],
                    metrics='accuracy' ,
                    model_name='activity')

plot_train_and_val(train = history.history['mobility_score_accuracy'],
                    val = history.history['val_mobility_score_accuracy'],
                    metrics='accuracy' ,
                    model_name='mobility_score')