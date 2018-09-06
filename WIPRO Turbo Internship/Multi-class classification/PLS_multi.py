from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import glob
import os
import numpy as np

def Shuffle (order,arr) :
	shuf = zip(order,arr)
	shuf.sort()
	data = [x for (y,x) in shuf]
	return data
	
path = "/home/WIPRO Turbo Internship/Datasets/Spices/"

data,labels = [],[]


for file_name in glob.glob(path + "*.csv") :
#for file_name in glob.glob("*.csv") :
	intensity = []
	lines = 0
	with open(file_name, "r") as filestream :
#		print file_name
		for line in filestream :
			lines += 1
			if (lines > 20) :
				currentline = line.split(",")
		#		wavelength.append(float(currentline[0]))
			    	intensity.append(float(currentline[1]))
		data.append(intensity)
		file_name = os.path.basename(file_name)
		tag = [0] * 19
#		name = file_name[len(path):]		
		if (file_name[:6] == 'Chilly') :
			tag[0] = 1
			labels.append(tag)
		elif (file_name[:6] == 'Coffee') :
			tag[1] = 1
			labels.append(tag)
		elif (file_name[:6] == 'Flatte') :
			tag[2] = 1
			labels.append(tag)
		elif (file_name[:6] == 'FriedG') :
			tag[3] = 1
			labels.append(tag)
		elif (file_name[:6] == 'GaramM') :
			tag[4] = 1
			labels.append(tag)
		elif (file_name[:6] == 'GramDa') :
			tag[5] = 1
			labels.append(tag)
		elif (file_name[:6] == 'GramFl') :
			tag[6] = 1
			labels.append(tag)
		elif (file_name[:6] == 'GreenG') :
			tag[7] = 1
			labels.append(tag)
		elif (file_name[:6] == 'IdliRi') :
			tag[8] = 1
			labels.append(tag)
		elif (file_name[:6] == 'JeeraC') :
			tag[9] = 1
			labels.append(tag)
		elif (file_name[:6] == 'MethiC') :
			tag[10] = 1
			labels.append(tag)
		elif (file_name[:6] == 'MoongD') :
			tag[11] = 1
			labels.append(tag)
		elif (file_name[:6] == 'Mustar') :
			tag[12] = 1
			labels.append(tag)
		elif (file_name[:6] == 'RaagiF') :
			tag[13] = 1
			labels.append(tag)
		elif (file_name[:6] == 'SonaMa') :
			tag[14] = 1
			labels.append(tag)
		elif (file_name[:6] == 'TeaCol') :
			tag[15] = 1
			labels.append(tag)
		elif (file_name[:6] == 'ToorDa') :
			tag[16] = 1
			labels.append(tag)
		elif (file_name[:6] == 'Turmer') :
			tag[17] = 1
			labels.append(tag)
		elif (file_name[:6] == 'UradDa') :
			tag[18] = 1
			labels.append(tag)
			
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.40, random_state = 987, stratify = labels)

#Pre-processing...
scaler = preprocessing.StandardScaler().fit(X_train)								#The transformation to be used on training and testing sets
X_train_scaled = scaler.transform(X_train)									#Standardising the training data (mean = 0,variance = 1) along each dimension
X_test_scaled = scaler.transform(X_test)									#Standardising the testing data

pls = PLSRegression(n_components = 20)
pls.fit(X_train,y_train)
pred = pls.predict(X_test)
#print pred
print pls.get_params
print pls.score(X_test,y_test)
#print "Actual : ",test_labels
result = np.zeros_like(pred)
result[np.arange(len(pred)), pred.argmax(1)] = 1
#print "Prediction : ",result
#print result - test_labels
count,corr = 0,0
actual = np.zeros_like(pred)
for i in range(0,len(y_test)) :
	count += 1
	if (y_test[i].index(1) == list(result[i]).index(1)) :
		corr += 1
print float(corr)/float(count)
#print result
#print actual
#print result - actual
#print float(len(y_test) - np.count_nonzero(result - actual))/float(len(y_test))
#print np.shape(result)
