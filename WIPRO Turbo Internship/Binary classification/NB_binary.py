import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing

files = ['ExpiredDigene1.csv','ExpiredDigene2.csv','ExpiredDigene3.csv','ExpiredDigene4.csv','NewDigene2.csv','NewDigene3.csv','NewDigene4.csv','NewDigene5.csv','NewDigene6.csv','NewDigene7.csv','NewDigene8.csv','NewDigene9.csv','NewDigene10.csv']
path = "/home/WIPRO Turbo Internship/Datasets/Digene/DigeneData/"

data = []									#Different sample spectra
labels = []									#Labels : 0 = Expired, 1 = New

for f in files :
	name = path + f
#	name = f
	intensity = []
	with open(name, "r") as filestream :
		for line in filestream :
			currentline = line.split(",")
		    	intensity.append(float(currentline[1]))
		intensity = intensity[:-2]
		data.append(intensity)
		if (f[:3] == 'Exp') :
			labels.append(0)
		else :
			labels.append(1)

#Converting to matrices...
data = np.array(data)
#labels = np.array(labels).reshape(len(files),1)

#Training and testing data split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 987, stratify = labels)

#Pre-processing...
scaler = preprocessing.StandardScaler().fit(X_train)								#The transformation to be used on training and testing sets
X_train_scaled = scaler.transform(X_train)									#Standardising the training data (mean = 0,variance = 1) along each dimension
X_test_scaled = scaler.transform(X_test)									#Standardising the testing data


#print np.unique(y_train)
clf = GaussianNB()
clf.partial_fit(X_train,y_train,np.unique(y_train))

pred = clf.predict(X_test)
pred = clf.predict(X_test)
print "R^2 score : ",r2_score(y_test,pred)									#Between -1 and 1, with -1 being bad model and 1 being accurate model

count, corr = 0,0
for i in range(0,len(y_test)) :
	count += 1
	if (float(y_test[i]) == float(pred[i])) :
		corr += 1

print "Accuracy : ",float(corr)/float(count)
