from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

files = ['ExpiredDigene1.csv','ExpiredDigene2.csv','ExpiredDigene3.csv','ExpiredDigene4.csv','NewDigene2.csv','NewDigene3.csv','NewDigene4.csv','NewDigene5.csv','NewDigene6.csv','NewDigene7.csv','NewDigene8.csv','NewDigene9.csv','NewDigene10.csv']
path = "/home/WIPRO Turbo Internship/Datasets/Digene/DigeneData/"

data = []
labels = []

for f in files :
	name = path + f
#	name = f
	intensity = []
	with open(name, "r") as filestream :
		for line in filestream :
			currentline = line.split(",")
	#		wavelength.append(float(currentline[0]))
		    	intensity.append(float(currentline[1]))
		intensity = intensity[:-2]
		data.append(intensity)
		if (f[:3] == 'Exp') :
			labels.append([1,0])
		else :
			labels.append([0,1])
			
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.30, random_state = 987, stratify = labels)

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
