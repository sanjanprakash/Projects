import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
import glob
import os

path = "/home/WIPRO Turbo Internship/Datasets/Spices/"

data,labels = [],[]

for file_name in glob.glob(path + "*.csv") :
#for file_name in glob.glob("*.csv") :
	intensity = []
	lines = 0
	with open(file_name, "r") as filestream :
		for line in filestream :
			lines += 1
			if (lines > 20) :
				currentline = line.split(",")
			    	intensity.append(float(currentline[1]))
		data.append(intensity)
		file_name = os.path.basename(file_name)
#		name = file_name[len(path):]		
		if (file_name[:6] == 'Chilly') :
			labels.append(0)
		elif (file_name[:6] == 'Coffee') :
			labels.append(1)
		elif (file_name[:6] == 'Flatte') :
			labels.append(2)
		elif (file_name[:6] == 'FriedG') :
			labels.append(3)
		elif (file_name[:6] == 'GaramM') :
			labels.append(4)
		elif (file_name[:6] == 'GramDa') :
			labels.append(5)
		elif (file_name[:6] == 'GramFl') :
			labels.append(6)
		elif (file_name[:6] == 'GreenG') :
			labels.append(7)
		elif (file_name[:6] == 'IdliRi') :
			labels.append(8)
		elif (file_name[:6] == 'JeeraC') :
			labels.append(9)
		elif (file_name[:6] == 'MethiC') :
			labels.append(10)
		elif (file_name[:6] == 'MoongD') :
			labels.append(11)
		elif (file_name[:6] == 'Mustar') :
			labels.append(12)
		elif (file_name[:6] == 'RaagiF') :
			labels.append(13)
		elif (file_name[:6] == 'SonaMa') :
			labels.append(14)
		elif (file_name[:6] == 'TeaCol') :
			labels.append(15)
		elif (file_name[:6] == 'ToorDa') :
			labels.append(16)
		elif (file_name[:6] == 'Turmer') :
			labels.append(17)
		elif (file_name[:6] == 'UradDa') :
			labels.append(18)

data = np.array(data)

#Training and testing data split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.30, random_state = 987, stratify = labels)

#Pre-processing...
scaler = preprocessing.StandardScaler().fit(X_train)								#The transformation to be used on training and testing sets
X_train_scaled = scaler.transform(X_train)									#Standardising the training data (mean = 0,variance = 1) along each dimension
X_test_scaled = scaler.transform(X_test)									#Standardising the testing data

clf = GaussianNB()
clf.partial_fit(X_train,y_train,np.unique(y_train))

pred = clf.predict(X_test)

print "R^2 score : ",r2_score(y_test,pred)

count, corr = 0,0
for i in range(0,len(y_test)) :
	count += 1
	if (float(y_test[i]) == float(pred[i])) :
		corr += 1

print "Accuracy : ",float(corr)/float(count)
