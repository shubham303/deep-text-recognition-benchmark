#!/usr/bin/env python
# coding: utf-8

# In[1]:
import csv
import os
import json
import glob

import cv2
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
data={}
with open('Prediction_English_FT.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			data[row[0]]=row[1:]



# In[5]:


gts = []

gt_texts = open(gt_path).readline().split("\i")
for i, gt in enumerate(gt_texts):
	gt = gt.split("\t")
	img = "{}{}".format(image_path.replace("images/", ""),gt[0])
	text = gt[1]
	data[img].append(text)


def levenshteinDistance(actual, prediction):
	s1 = actual
	s2 = prediction
	if len(s1) > len(s2):
		s1, s2 = s2, s1
	
	distances = range(len(s1) + 1)
	for i2, c2 in enumerate(s2):
		distances_ = [i2 + 1]
		for i1, c1 in enumerate(s1):
			if c1 == c2:
				distances_.append(distances[i1])
			else:
				distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
		distances = distances_
	return distances[-1]




from sklearn.metrics import accuracy_score


actual = [ v[0].lower() for v in data.values()]
prediction = [v[2].lower() for v in data.values()]
test_accuracy = accuracy_score(actual,prediction)

# In[15]:


print(test_accuracy)

# In[16]:



# In[17]:


def visualize_misclassifications(split, num_samples=20, max_edit_distance=0):
	for k, d in data.items():
		img = cv2.imread(k)
		if not d[0].lower()== d[2].lower():
			img_name = "result/incorrect-predicted{}".format(k[k.rfind("/"):])
			print("predicted :{}  actual:{}".format(d[0].lower(), d[2].lower()))
		else:
			img_name = "result/correctly-predicted{}".format(k[k.rfind("/"):])
		cv2.imwrite(img_name, img)





visualize_misclassifications('Test')

# In[ ]:




