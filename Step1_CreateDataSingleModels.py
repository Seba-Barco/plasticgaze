#!/usr/bin/env python
# coding: utf-8

# In[10]:


# create feature data for single class models

import os, sys, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

root = ""

fileExt = "_"
class_names = ['0', 'PE', 'PET', 'PP', 'PS', 'PVC', 'CuPC']
sourceDir = root + "Dataset/"
featureDir = root + "Features/"

# iterate over all directories
for d_test in os.listdir(sourceDir):
    with open(sourceDir + '/' +  d_test + '/' + d_test + '.json', 'r') as f:
        data = json.load(f)
        for classID in [1, 2, 3, 4, 5, 6]:
            print("Generate Data:", d_test, classID)
            try:
                del x_train
            except:
                pass

            for e in data:
                filename = sourceDir + '/' +  d_test + '/spectrum/' + str(e['id']) + '.json'
            
                try:
                    with open(filename, "r") as sf:
                        
                        ll_t = np.array(json.load(sf))
                        ll_t[np.isnan(ll_t)] = 0
                    
                        
                        X_train = np.array(range(0,len(ll_t)))
                        y_train = np.array(ll_t)

                        ll = np.gradient(y_train, axis=0)

                        peaks = {
                            "1": [1052, 1120, 1295, 1440], 
                            "2": [1100, 1278, 1620, 1735],  
                            "3": [785, 820, 1145, 1328, 1458], 
                            "4": [990, 1020, 1181, 1587, 1607],  
                            "5": [612, 663, 1435],  
                            "6": [660, 727, 1140, 1341, 1535]  
                        }
                        
                        area = 35
                        ind = []
                        for i in peaks[str(classID)]:
                            ind = ind + list(range(int(i/2) - area, int(i/2) + area + 1))

                        ll1 = ll[ind]
                        ll1 = ll1.reshape(-1,1)
                        scaler = StandardScaler()
                        scaler.fit(ll1)
                        ll1 = scaler.transform(ll1)
                        ll1 = ll1.flatten()
                        
                        if not 'x_train' in locals():
                            x_train = np.zeros([0, len(ll1)])

                        x_train = np.append(x_train, [ll1], axis=0)
                        
                except Exception as inst:
                    print("Error:", inst)  
                    

            print("SHAPE Train: ", x_train.shape)
            file_path = featureDir + d_test
            
            Path(file_path).mkdir(parents=True, exist_ok=True)
            #now = datetime.datetime.now()
            #print (now.strftime("%Y-%m-%d %H:%M:%S"))
            np.savez(open(featureDir + d_test + '/'  + d_test + fileExt + 'class' + str(classID) + '.npz', 'wb'), x_train=x_train)#, y_train=y_train)

print("the end")


# In[ ]:




