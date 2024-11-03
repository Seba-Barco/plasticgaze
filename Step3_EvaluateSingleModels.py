#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, json
import h5py # read mapx file
import matplotlib.pyplot as plt
from matplotlib import image, gridspec
from sklearn.model_selection import KFold
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Disable TensorFlow warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[ ]:


# evaluate based on prob 6 models
#%matplotlib notebook

import time

root = ""

featureDir = root + "Features/"
modelDir = root + "Models/"
jsonDir = root + "Dataset/"

class_names = ['0', 'PE', 'PET', 'PP', 'PS', 'PVC', 'CuPC']
complete_names = ['Polietileno', 'Tereftalato de Polietileno', 'Polipropileno', 'Poliestireno', 'Cloruro de Polivinilo', 'Sin Identificar']

stat = {}

for classInd in [1,2,3,4,5]:
    stat[classInd] = {}
    stat[classInd]['precision'] = []
    stat[classInd]['recall'] = []
 
c = 0
nodes = 32
bias = 1
checks = []
for classInd in [1, 2,3 ,4 , 5]:
    for thresh in [ 0.5]: 
        #print("\n--------------------------------------------")
        print("Class: ", classInd, class_names[classInd], thresh)

        for version in ['_with_weights_class']: 
            fn_sum = 0
            fp_sum = 0
            tp_sum = 0
            instSum = 0
            nrTests = 0

            for dirName in os.listdir(featureDir): 
                try:
                    new_model = tf.keras.models.load_model(modelDir + dirName + '/'  + dirName + version + str(classInd) +'_nodes' + str(nodes))
                    npzfile = np.load(open(featureDir + dirName + '/'  + dirName + '_class' + str(classInd) + '.npz', 'rb'), allow_pickle=True)
                except:
                    print("Error:", dirName)
                    continue

                with open(jsonDir + '/' +  dirName + '/' + dirName + '.json', 'r') as f:
                    jsonData = json.load(f)


                def findIndex(x):
                    ind = 1 if class_names[classInd] in x['label'] else 0
                    return ind


                y_test = list(map(lambda x: findIndex(x), jsonData))
                y_test = np.array(y_test)

                c = c + 1
                x_test = npzfile['x_train']

                predictions = new_model.predict(x_test)
                combined = sorted(zip(predictions, jsonData, y_test), key=lambda pair: pair[0][0], reverse=True)
                
                tp_new = 0
                fp_new = 0
                number_new = 0

                for ele in combined:
                    if (ele[0] < thresh):
                        #print("Prob:", ele[0])
                        break
                        
                    number_new = number_new + 1
                    if ele[2] == 1:
                        tp_new = tp_new + 1
                    else:
                        checks.append({"id": ele[1]['id'], "dirName": dirName, "prob": ele[0]})
                        fp_new = fp_new + 1

                fn_new = np.sum(y_test == 1) - tp_new

                #print(version, " FN:", fn_new , " TP:", tp_new, " FP:", fp_new, dirName, "tests: ", number_new, " prob:", ele[0])
                fn_sum = fn_sum + fn_new
                fp_sum = fp_sum + fp_new
                tp_sum = tp_sum + tp_new
                instSum = instSum + len(y_test)
                nrTests = nrTests + number_new
                
            if (tp_sum + fn_sum)> 0:
                recall = tp_sum / (tp_sum + fn_sum)
                precision = tp_sum / (tp_sum + fp_sum)

                print("Tests:", nrTests, " False Negatives:", fn_sum, " False Positives:", fp_sum, " True Positives:", tp_sum )
                stat[classInd]['recall'].append(tp_sum / (tp_sum + fn_sum))
                stat[classInd]['precision'].append(tp_sum/(tp_sum + fp_sum))

                print(complete_names[classInd - 1])  # Print the complete name of the plastic
                print("Recall: {:.2f}".format(recall * 100))  # Format recall to 2 decimal places
                print("Precision: {:.2f}".format(precision * 100))  # Format precision to 2 decimal places
                print("-----------------------------")
    
            else:
                print("sum is 0")
            

print("Analysis Complete")


# In[ ]:




