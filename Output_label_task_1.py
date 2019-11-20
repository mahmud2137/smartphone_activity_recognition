import numpy as np 
import pandas as pd 

train_dir = 'har_dataset/train/'
train_labels = pd.read_csv(train_dir+'y_train.txt', names=['label'])
test_dir = 'har_dataset/test/'
test_labels = pd.read_csv(test_dir+'y_test.txt', names=['label'])

train_labels = train_labels.label.values
test_labels = test_labels.label.values

'''
1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING
'''

def map_label_1(label):
    label_1_map = {1:'WALKING',
                2:'WALKING',
                3:'WALKING',
                4:'SITTING',
                5:'STANDING',
                6:'LAYING'}
    label = pd.Series(label)
    label_1 = label.map(label_1_map)
    return label_1.values

def map_mobility_score(label):
    '''
    Laying 0
    Sitting 1
    Standing 1
    Walking Downstairs 2
    Walking 3
    Walking Upstairs 5
    '''
    label = pd.Series(label)
    mobility_score_map = {6:0,4:1,5:1,3:2,1:3,2:5}
    mobility_score = label.map(mobility_score_map)
    return mobility_score.values

map_mobility_score(train_labels)

if __name__ == "__main__":
    train_labels_1 = map_label_1(train_labels)
    print(train_labels_1)
    train_mobility_score = map_mobility_score(train_labels)
    print(train_mobility_score)