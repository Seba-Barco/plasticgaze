#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import regularizers
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


# In[4]:


# evaluate multi label model (1 Model)
root = "please specify" 

featureDir = "FeaturesAnonym/"
modelDir = "DataModels/"
jsonDir = "DataAnonym/"

class_names = ['0', 'PE', 'PET', 'PP', 'PS', 'PVC', 'CuPC']
fileExt = "_all"

for modelConf in ['m0']:
    for classInd in [1, 2, 3, 4, 5, 6]:
        for thresh in [0.5]:
            print(classInd, modelConf, " Thresh:", thresh)
            fn_sum = 0
            fp_sum = 0
            tp_sum = 0
            instSum = 0
            nrTests = 0
            for d_test in os.listdir(featureDir):
     
                try:
                    new_model = tf.keras.models.load_model(modelDir + d_test + '/'  + d_test + fileExt + '_' +  modelConf)
                    npzfile = np.load(open(featureDir + d_test + '/'  + d_test + fileExt  + '.npz', 'rb'))
                    with open(jsonDir + '/' +  d_test + '/' + d_test + '.json', 'r') as f:
                        jsonData = json.load(f)
                except Exception as inst:
                    print(type(inst)) 
                    print(inst)
                    #break;
                    continue

                x_train = npzfile['x_train']
                yhat = new_model.predict(x_train)

                def findIndex(x):
                    ind = 1 if class_names[classInd] in x['label'] else 0
                    return ind

                y_test = list(map(lambda x: findIndex(x), jsonData))
                y_test = np.array(y_test)

                x_test = npzfile['x_train']
                
                
                predictions = new_model.predict(x_test)
                combined = sorted(zip(predictions, jsonData, y_test), key=lambda pair: pair[0][classInd], reverse=True)
                
                tp_new = 0
                fp_new = 0
                number_new = 0

                for ele in combined:
                    if (ele[0][classInd] < thresh):
                        #print("Prob:", ele[0])
                        break
                        
                    number_new = number_new + 1
                    if ele[2] == 1:
                        tp_new = tp_new + 1
                    else:
                        #checks.append({"id": ele[1]['id'], "dirName": dirName, "prob": ele[0][classInd]})
                        fp_new = fp_new + 1



                fn_new = np.sum(y_test == 1) - tp_new
                #print(d_test, " TP:", tp_new, " FN:", fn_new ,  " FP:", fp_new, "tests: ", number_new) #, " prob:", ele[0])

                fn_sum = fn_sum + fn_new
                fp_sum = fp_sum + fp_new
                tp_sum = tp_sum + tp_new
                instSum = instSum + len(y_test)
                nrTests = nrTests + number_new

            if (tp_sum + fn_sum)> 0:
        
                print('\n\n--------------------------')
                print("SUMMARY for: ", " classInd:", classInd, " (in %)", modelConf)
                print("Recall: ", tp_sum / (tp_sum + fn_sum), " Tested Instances: ", nrTests/instSum)
                print("Precision: ", tp_sum/(tp_sum + fp_sum))
                print("Instances:", instSum, " Tests:", nrTests, " FN (sum):", fn_sum, " FP (sum):", fp_sum, " TP (sum):", tp_sum )

                print(round(100 * tp_sum / (tp_sum + fn_sum), 1), round(100 * tp_sum/(tp_sum + fp_sum), 1))

print("the end")
        



# In[ ]:




