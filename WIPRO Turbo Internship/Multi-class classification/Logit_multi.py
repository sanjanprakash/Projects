import numpy as np
import glob
import os
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.externals import joblib

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
labels = np.array(labels).reshape(len(data),1)
whole = np.hstack([labels,data])

col_heads = ['labels']
for i in range(0,len(whole[0]) - 1) :
	col_heads.append(str(i))
col_heads = np.array(col_heads).reshape(1,len(whole[0]))

whole = np.vstack([col_heads,whole])

row_heads = ['']
for i in range(0,len(whole) - 1) :
	row_heads.append(str(i))
row_heads = np.array(row_heads).reshape(len(whole),1)

whole = np.hstack([row_heads,whole])
df = pd.DataFrame(data = whole[1:,1:],index = whole[1:,0],columns = whole[0,1:])				#Dataframe holding the complete data

#Splitting the data into labels and raw data...
y = df[df.columns[0]]
X = df.drop('labels',axis = 1)
y = pd.to_numeric(y)
X = X.astype(float)

#Training and testing data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 987, stratify = y)

#Pre-processing...
#scaler = preprocessing.StandardScaler().fit(X_train)								#The transformation to be used on training and testing sets
#X_train_scaled = scaler.transform(X_train)									#Standardising the training data (mean = 0,variance = 1) along each dimension
#X_test_scaled = scaler.transform(X_test)									#Standardising the testing data

pipeline = make_pipeline(preprocessing.StandardScaler(),LogisticRegression(max_iter = 10000,random_state = 123,class_weight = 'balanced'))
#print pipeline.get_params()											#Listing out the hyperparameters that can be fine-tuned
hyperparameters = {'logisticregression__C' : [1e-3,1e-2,1e-1,1e0],'logisticregression__intercept_scaling' : [0.05,0.1,0.5,1.0],'logisticregression__tol' : [1e-8,1e-6,1e-4,1e-2,1e0],'logisticregression__solver' : ['newton-cg','sag','lbfgs']}	#Choosing the hyperparameters that are to be tuned
clf = GridSearchCV(pipeline,hyperparameters,cv  = 4)

clf.fit(X_train.values,y_train.values)
print clf.best_params_												#To view the optimum values of the hyperparameters

pred = clf.predict(X_test)
print "R^2 score : ",r2_score(y_test,pred)											#Between -1 and 1, with -1 being bad model and 1 being accurate model

count, corr = 0,0
for i in range(0,len(y_test)) :
	count += 1
	if (float(y_test[i]) == float(pred[i])) :
		corr += 1

print "Accuracy : ",float(corr)/float(count)
joblib.dump(clf, 'Logit_multi_classifier.pkl')									#Storing the model, to be used later

#To load the model and test it on another test set
#clf2 = joblib.load('Logit_multi_classifier.pkl')
#clf2.predict(test_samples)
#{'logisticregression__solver': 'newton-cg', 'logisticregression__tol': 1e-08, 'logisticregression__intercept_scaling': 0.05, 'logisticregression__C': 1.0}
