#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os, sys
import shutil

sourceDir = "//hobbes/Mikroplastik/# InOut/Andreas Zinnen/data_paper/"
targetDir = "//hobbes/Mikroplastik/# InOut/Andreas Zinnen/DataAnonym/"

datadirs = os.listdir( sourceDir )
counter = 0
s = '['
for fileDir in datadirs:
    s = s + '"' + "dir" + str(counter) + '",'
    #print('{"' + fileDir, "dir" + str(counter))
    #shutil.copy(sourceDir + fileDir + '/' + fileDir + '.json', targetDir + "dir" + str(counter) + '/' + "dir" + str(counter) + '.json')
    #os.mkdir(targetDir + "dir" + str(counter))
    #shutil.copytree(sourceDir + fileDir + '/spectrum', targetDir + "dir" + str(counter) + "/spectrum", dirs_exist_ok=True)
    counter = counter + 1

print (s)
print("Ende")



# In[ ]:




