#!/usr/bin/env python
# coding: utf-8

# In[14]:


# mlp for multi-label classification
from numpy import mean
import numpy as np
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
import os, sys, json
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras import regularizers
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root = ""

featureDir = root + "Features/"
modelDir = root + "DataModels/"
jsonDir = root + "Dataset/"
fileExt = "_all"
class_names = ['0', 'PE', 'PET', 'PP', 'PS', 'PVC', 'CuPC']

# sample mapping contains all coherent samples that are not to be taken into account during training. 
samplemapping = [
    ['dir4', 'dir5'], 
    ['dir61', 'dir7', 'dir8'],
    ['dir9', 'dir10', 'dir11'],
    ['dir12', 'dir13', 'dir14'],
    ['dir15', 'dir16', 'dir17'],
    ['dir18', 'dir19', 'dir20'],
    ['dir42', 'dir43'],
    ['dir45', 'dir46']
]

for modelConf in ['m0']:
    for d_test in os.listdir(featureDir):
        try:
            new_model = tf.keras.models.load_model(modelDir + d_test + '/'  + d_test + fileExt + '_' +  modelConf)
            print("exists: ", d_test)
            continue
        except:
            pass
        
        print("train: ", d_test, datetime.datetime.now())
        
        # find out if d_test should be ignored
        ignore = []
        for e in samplemapping:
            if d_test in e:
                ignore = e

        try:
            del x_train
        except:
            pass

        try:
            del y_train
        except:
            pass

        for d_train in os.listdir(featureDir):
            if d_train in ignore:
                print("ignore: ", d_test, ignore)
                continue
            
            npzfile = np.load(open(featureDir + d_train + '/'  + d_train + fileExt  + '.npz', 'rb'))

        
            if not 'x_train' in locals():
                x_train = npzfile['x_train']
            else:
                x_train = np.concatenate((x_train, npzfile['x_train']))
            
            with open(jsonDir + '/' +  d_train + '/' + d_train + '.json', 'r') as f:
                data = json.load(f)
                labels = np.zeros((len(data), 7))

                counter = 0
                for e in data:
                    anno = list(map(lambda a: 1 if a in e['label'] else 0, class_names))
                    labels[counter, :]= anno
                    counter = counter + 1

            if not 'y_train' in locals():
                y_train = labels
            else:
                y_train = np.concatenate((y_train, labels))
            
        t_0 = sum(map(lambda x: int(x[0]), y_train))
        t_1 = sum(map(lambda x: int(x[1]), y_train))
        t_2 = sum(map(lambda x: int(x[2]), y_train))
        t_3 = sum(map(lambda x: int(x[3]), y_train))
        t_4 = sum(map(lambda x: int(x[4]), y_train))
        t_5 = sum(map(lambda x: int(x[5]), y_train))
        t_6 = sum(map(lambda x: int(x[6]), y_train))
        ss = t_0 + t_1 + t_2 + t_3 + t_4 + t_5 + t_6
        weights = {0: 1/t_0 * ss / 2, 1: 1/t_1 * ss / 2, 2: 1/t_2 * ss / 2, 3: 1/t_3 * ss / 2, 4: 1/t_4 * ss / 2, 5: 1/t_5 * ss / 2, 6: 1/t_6 * ss / 2}
        print("?? weights", weights)
        if modelConf == "m1" or modelConf == "m2":
            model = keras.Sequential([
                keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001), input_shape=(x_train.shape[-1],)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(7, activation='sigmoid'),
            ])
        elif modelConf == 'm3' or modelConf == "m4":
            model = keras.Sequential([
                keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001), input_shape=(x_train.shape[-1],)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(7, activation='sigmoid'),
            ])
            
        elif modelConf == 'm5':
            model = keras.Sequential([
                keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01), input_shape=(x_train.shape[-1],)),
                keras.layers.Dropout(0.7),
                keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01)),
                keras.layers.Dropout(0.7),
                keras.layers.Dense(7, activation='sigmoid'),
            ])
            
        elif modelConf == "m0":
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001), input_shape=(x_train.shape[-1],)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(7, activation='sigmoid'),
            ])
        
        elif modelConf == "m00":
            model = keras.Sequential([
                keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001), input_shape=(x_train.shape[-1],)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(7, activation='sigmoid'),
            ])
            
        optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipvalue=0.5)
        
        model.compile(
          optimizer=optimizer,
          loss=tf.keras.losses.BinaryCrossentropy()
        )
        
        #print(x_train.shape, y_train.shape)
        x_train[np.isnan(x_train)] = 0
        
        if modelConf == 'm0' or modelConf == 'm00' or modelConf == 'm2' or modelConf == 'm4'or modelConf == "m5": 
            model.fit(x_train, y_train, verbose=0, epochs=1500, batch_size=100000, class_weight=weights)
        else:
            model.fit(x_train, y_train, verbose=0, epochs=1500, batch_size=10000)
        
        
        file_path = modelDir + d_test   
        Path(file_path).mkdir(parents=True, exist_ok=True)
        model.save(modelDir + d_test + '/'  + d_test + fileExt + '_'+ modelConf)
 
print("The End")


# In[ ]:




