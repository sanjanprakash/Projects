import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.externals import joblib

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
labels = np.array(labels).reshape(len(files),1)
whole = np.hstack([labels,data])

#Converting the complete dataset into dataframes...
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 987, stratify = y)

#Pre-processing...
#scaler = preprocessing.StandardScaler().fit(X_train)								#The transformation to be used on training and testing sets
#X_train_scaled = scaler.transform(X_train)									#Standardising the training data (mean = 0,variance = 1) along each dimension
#X_test_scaled = scaler.transform(X_test)									#Standardising the testing data

pipeline = make_pipeline(preprocessing.StandardScaler(),GradientBoostingClassifier(random_state = 987))
#print pipeline.get_params().keys()											#Listing out the hyperparameters that can be fine-tuned
hyperparameters = {'gradientboostingclassifier__loss' : ['exponential','deviance'],'gradientboostingclassifier__learning_rate' : [0.01,0.1], 'gradientboostingclassifier__n_estimators' : range(100,50,-10),'gradientboostingclassifier__max_depth' : range(3,1,-1),'gradientboostingclassifier__max_features' : ['sqrt','auto','log2',None]}	#Choosing the hyperparameters that are to be tuned

clf = GridSearchCV(pipeline,hyperparameters,cv  = 3)

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
joblib.dump(clf,'GradientBoost_binary_classifier.pkl')									#Storing the model, to be used later

#To load the model and test it on another test set
#clf2 = joblib.load('GradientBoost_binary_classifier.pkl')
#clf2.predict(test_samples)
