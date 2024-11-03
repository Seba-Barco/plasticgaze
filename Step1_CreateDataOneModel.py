#!/usr/bin/env python
# coding: utf-8

# In[8]:


# create feature data for multi class models

import os, sys, json
import datetime
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

root = ""

class_names = ['0', 'PE', 'PET', 'PP', 'PS', 'PVC', 'CuPC']
sourceDir = root + "Dataset/"
featureDir = root + "Features/"

fileExt = "_all"

# iterate over all directories
for d_test in os.listdir(sourceDir):
    print(d_test, datetime.datetime.now())
    with open(sourceDir + '/' +  d_test + '/' + d_test + '.json', 'r') as f:
        data = json.load(f)
        x_train = np.zeros((len(data), 611))
        
        counter = 0
        for e in data:
            filename = sourceDir + '/' +  d_test + '/spectrum/' + str(e['id']) + '.json'
            with open(filename, "r") as sf:
                ll_t = np.array(json.load(sf))
                ll = np.gradient(ll_t, axis=0)
                # obtain necessary area for all classes
                ll = ll[int(562/2):int(1785/2)]
                ll_t = ll_t[int(562/2):int(1785/2)]
                
                ll[np.isnan(ll)] = 0
                ll = ll.reshape(-1,1)
                scaler = StandardScaler()
                scaler.fit(ll)
                ll = scaler.transform(ll)
                ll = ll.flatten()
                
                x_train[counter, :] = ll

            counter= counter + 1
            
        file_path = featureDir + d_test   
        Path(file_path).mkdir(parents=True, exist_ok=True)
        np.savez(open(featureDir + d_test + '/'  + d_test + fileExt + '.npz', 'wb'), x_train=x_train)
        
print("Feature data created")