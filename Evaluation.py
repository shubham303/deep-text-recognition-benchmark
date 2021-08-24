#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import glob

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[2]:


prediction_path = "Prediction_English_FT.csv"
gt_path = "/home/shubham/Documents/MTP/datasets/2/test/english/gt.txt"
image_path = "/home/shubham/Documents/MTP/datasets/2/test/english/images/"


# In[3]:


# prediction_path = "predictions/Prediction_TPS_ResNet_BiLSTEM_Attn_english_back_gen.csv"
# gt_path = "../AutoID_data/Voter-Hi_old/Back-gen/Crops_english/Texts/"
# image_path = "../AutoID_data/Voter-Hi_old/Back-gen/Crops_english/Images/"


# In[4]:


data = pd.read_csv(prediction_path)
data = data.loc[data['filename'] != 'image_path']
data.head()


# In[5]:


gts = []

gt_texts = open(gt_path).readline().split("/i")
for gt in gt_texts:
    gt = gt.split("\t")
    img =gt[0]
    text = gt[1]
    gts.append({})
for idx in data.index:
    image_name = data['filename'][idx].strip().split("/")[-1]
    gt_filepath = os.path.join(gt_path, image_name.replace("jpg", "txt"))
    with open(gt_filepath, 'r') as f:
        gt = f.read()
    gt_text = gt.split("_")[0]
    gts.append(gt_text)


# In[6]:


data['actual'] = gts


# In[7]:


data.head()


# In[8]:


len(data)


# In[9]:


data['actual_length'] = data['actual'].apply(len)


# In[10]:


data['prediction_length'] = data['prediction'].apply(len)


# In[11]:


def levenshteinDistance(row):
    
    s1 = row['actual']
    s2 = row['prediction']
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# In[12]:


data['edit_distance'] = data.apply(levenshteinDistance, axis=1)


# In[13]:


data.head()


# In[14]:


from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(data['actual'], data['prediction'])


# In[15]:


print(test_accuracy)


# In[16]:


accuracy_df = []
for max_edit_distance in range(4+1):
    test_accuracy = (data['edit_distance'] <= max_edit_distance).sum() / len(data)
    accuracy_df.append([max_edit_distance, test_accuracy])
columns=['Max-Edit-Distance', 'Test-Accuracy']
accuracy_df = pd.DataFrame(accuracy_df, columns=columns)
accuracy_df.to_csv("Accuracy.csv", index=False)
accuracy_df


# In[17]:


def visualize_classifications(split, num_samples=20, max_edit_distance=0):
    
    result = data  
    mask = result['edit_distance'] == max_edit_distance
    result = result[mask].sample(n = num_samples)
    for row_id, row in result.iterrows():
        info = 'Actual:', row['actual'], 'Prediction:', row['prediction']
        print(info)
        print(row["filename"])
        image_fp = os.path.join(image_path, row['filename'].strip().split("/")[-1])
        plt.imshow(Image.open(image_fp))
        plt.axis(False)
        plt.show()


# In[18]:


visualize_classifications('Test')


# In[19]:


def visualize_misclassifications(split, num_samples=20, max_edit_distance=1):
    
    result = data  
    mask = result['edit_distance'] >= max_edit_distance
    result = result[mask].sample(n = num_samples)
    for row_id, row in result.iterrows():
        info = 'Actual:', row['actual'], 'Prediction:', row['prediction']
        print(info)
        print(row["filename"])
        image_fp = os.path.join(image_path, row['filename'].strip().split("/")[-1])
        plt.imshow(Image.open(image_fp))
        plt.axis(False)
        plt.show()


# In[20]:


visualize_misclassifications('Test')


# In[ ]:




