#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
import tensorflow as tf
#from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, json
import h5py # aread mapx file
import matplotlib.pyplot as plt
from matplotlib import image, gridspec
from sklearn.model_selection import KFold
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from keras import regularizers
from sklearn.preprocessing import StandardScaler
import datetime


# In[3]:


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.FalseNegatives(name='fn'), 
]
    
def make_model(nodes, x_train, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

        if nodes == 64:
             model = keras.Sequential([
                keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001), input_shape=(x_train.shape[-1],)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1, activation='sigmoid',bias_initializer=output_bias),
            ])
        else:   
            model = keras.Sequential([
                keras.layers.Dense(nodes, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(x_train.shape[-1],)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1, activation='sigmoid',bias_initializer=output_bias),
            ])

    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipvalue=0.5)

    model.compile(
      optimizer=optimizer,
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=metrics)

    return model

def trainModelNew(bias, nodes, classInd, x_train, y_train,modelPath, EPOCHS=500, BATCH_SIZE=2048):

    all = sum(y_train == 0) + sum(y_train == 1)
    weights = {0: 1/sum(y_train == 0) * all/2,1: 1/sum(y_train == 1) * all/2}
    x_train[np.isnan(x_train)] = 0
    initial_bias = np.log([1/bias * np.sum(y_train==1)/np.sum(y_train==0)])
    output_bias = tf.keras.initializers.Constant(initial_bias)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    model = make_model(nodes, x_train, output_bias=initial_bias)
    model.fit(x_train, y_train, epochs=EPOCHS, verbose=0, batch_size=BATCH_SIZE, class_weight=weights)
    
    print("save model for ", classInd)
    model.save(modelPath)


# In[6]:


# train the model

root = ""

featureDir = "Features/"
targetDir = "Models/"
jsonDir = "Dataset/"
class_names = ['0', 'PE', 'PET', 'PP', 'PS', 'PVC', 'CuPC']

now = datetime.datetime.now()
print (now.strftime("%Y-%m-%d %H:%M:%S"))

prefix = '_with_weights_class'

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

for classInd in [1, 2, 3, 4, 5, 6]:
    for d_test in os.listdir(featureDir):
        ignore = []
        for e in samplemapping:
            if d_test in e:
                ignore = e
        
        for nodes in [32]:
            try:
                new_model = tf.keras.models.load_model(targetDir  + d_test + '/' + d_test + prefix + str(classInd) +'_nodes' + str(nodes))
                print("exists: ", d_test, classInd)
                continue
            except:
                pass

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

                npzfile = np.load(open(featureDir + d_train + '/'  + d_train + '_class' + str(classInd) + '.npz', 'rb'))

                if not 'x_train' in locals():
                    x_train = npzfile['x_train']
                else:
                    x_train = np.concatenate((x_train, npzfile['x_train']))

                with open(jsonDir + '/' +  d_train + '/' + d_train + '.json', 'r') as f:
                    jsonData = json.load(f)

                def findIndex(x):
                    ind = 1 if class_names[classInd] in x['label'] else 0
                    return ind

                y_cur = list(map(lambda x: findIndex(x), jsonData))
                y_cur = np.array(y_cur)

                if not 'y_train' in locals():
                    y_train = y_cur
                else:
                    y_train = np.concatenate((y_train, y_cur))

            trainModelNew(1, nodes, classInd, x_train, y_train, targetDir + d_test + '/'  + d_test + prefix + str(classInd) +'_nodes' + str(nodes), 1500, 100000) 
            now = datetime.datetime.now()
            print (now.strftime("%Y-%m-%d %H:%M:%S"))

print("the end")



# In[ ]:




