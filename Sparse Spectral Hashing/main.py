from sklearn.decomposition import SparsePCA as sp
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import *
# from imtools import *
import os
from PIL import Image
import glob


#-----Pre-processing the images from the dataset
def get_image(filename):
    img = cv2.imread(filename)  #Read image in BGR order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #Change to RGB order
    img = cv2.resize(img, (224, 224))  #Resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  #Change to (channel, height, width)
    img = img[np.newaxis, :]  #Extend to (example, channel, heigth, width)
    return img


#--------Load images from Dataset

# out=[]
# data_pos = np.array([get_image('/home/safwan/Desktop/bdi/Dataset/football/' + img) for img in os.listdir('/home/safwan/Desktop/bdi/Dataset/football/')])
# data_neg = np.array([get_image('/home/safwan/Desktop/bdi/Dataset/lion/' + img) for img in os.listdir('/home/safwan/Desktop/bdi/Dataset/lion/')])
# data_neg1 = np.array([get_image('/home/safwan/Desktop/bdi/Dataset/guitar/' + img) for img in os.listdir('/home/safwan/Desktop/bdi/Dataset/guitar/')])
# datas = np.array([get_image('/home/safwan/Desktop/bdi/Dataset/oxford/' + img) for img in os.listdir('/home/safwan/Desktop/bdi/Dataset/oxford/')])

# X = np.append(data_pos,data_neg,axis = 0)
# Y = np.append(data_neg1,datas,axis = 0)
# Z = np.append(X,Y,axis = 0)


import pandas as pd
import numpy as np
from sklearn.decomposition import SparsePCA




X = pd.read_csv('mylist.csv', delimiter = None)
X = X.drop(X.columns[0],axis = 1)
X = np.array(X)


#-----SPCA
pca = SparsePCA(n_components = 1024)
x = pca.fit_transform(X)

df = pd.DataFrame(x)
df.to_csv('n_components_full.csv')

import math


#-----Mapping from Euclidean Space to Hamming Space
#---- x  = matrix B mentioned in the paper in the Step 1 of SPCA

#------This function returns the j-th binary value z(i, j) for corresponding i-th image via computing delta_kj matrix(m,m)
#Given the i-th image x(i) in  R(d) in training set, assume its corresponding binary code
#is y(i) with range{-1, 1} in m dimensions and the value after mapping is z(-1, 1) in m dimensions.
#The j-th binary value z(i, j) and y(i) (binary coded value) is returned by this function :

def hamming_z(x):
	x = np.array(x)
	row,col = x.shape
	# print x
	lis = []


	epsilon = 1e-7
	delta_kj = np.zeros((col,col))
	col_min = []
	col_max = []
	lister = []
	for j in range(col):
		for i in xrange(row):
	 		lis.append(x[i][j])
		col_min.append(np.min(np.array(lis)))
		col_max.append(np.max(np.array(lis)))
		lis = []

	for j in range(col):
		f = col_max[j]
		e =  col_min[j]
		if e - f == 0:         #Handling overflow
			for k in range(col):
				delta_kj[j][k] = 1
		        lister.append([delta_kj[j][k],[j,k]])
		else:
			for k in range(col):
				delta_kj[j][k] = (1 - math.exp((-(epsilon ** 2) * 0.5 * (((k + 1) * math.pi) * 1.0/(f - e)) ** 2)))
		        lister.append([delta_kj[j][k],[j,k]])
	lister.sort()
	indexes = {}
	t = 0
	for l in range((col)):
		indexes[l] = lister[l]
	z = np.zeros((row,col))
	y = np.zeros((row,col))
	for u in range(row):
		for v in range(col):
			if (col_max[indexes[v][1][0]] - col_min[indexes[v][1][0]]) == 0:   #Handling overflow
				z[u][v] = 1
			else:
				z[u][v] = math.sin(math.pi/2 + ((indexes[v][1][1] * 11 * x[u][indexes[v][1][0]]) * 1.0 * math.pi/(col_max[indexes[v][1][0]] - col_min[indexes[v][1][0]])))

			if z[u][v] <= t:
				y[u][v] = 1
			else:
				y[u][v] = -1
	return z,y

X = pd.read_csv('n_components_full.csv', delimiter = None)
X = X.drop(X.columns[0], axis = 1)
X = np.array(X)
print X.shape,X
z,y = hamming_z(X)

df = pd.DataFrame(z)
# df = df.drop(df.columns[0], axis = 0)
df.to_csv('z_values_full.csv')

df = pd.DataFrame(y)
# df = df.drop(df.columns[0], axis = 1)
df.to_csv('y_values_full.csv')
