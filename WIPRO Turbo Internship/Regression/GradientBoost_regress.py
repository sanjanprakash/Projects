import numpy as np
import glob
import os
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.externals import joblib

reference_values = []
lines = 0
#with open("/home/sanjan/HSI/Biscuits/NutritionData.csv","r") as reference :
with open("/home/WIPRO Turbo Internship/Datasets/Biscuits/NutritionData.csv","r") as reference :
	for l in reference :
		lines += 1
		#2 - Calories, 3 - Total Fat, 4 - Saturated Fat, 5 - Poly Unsaturated Fat, 6 - Mono Unsaturated Fat, 7 - Cholestrol, 8 - Carbohydrates, 9 - Sugar, 10 - Protein 
		if (lines == 2) :
			currentline = l.split(",")
			for i in range(2,12) :
				reference_values.append(float(currentline[i]))

path = "/home/WIPRO Turbo Internship/Datasets/Biscuits/BiscuitsData/"

data,content_value = [],[]

for file_name in glob.glob("*.csv") :
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
		if (file_name[:9] == 'EliteMilk') :
			content_value.append(reference_values[0])
		elif (file_name[:12] == 'EliteElaichi') :
			content_value.append(reference_values[1])
		elif (file_name[:4] == 'Brit') :
			content_value.append(reference_values[2])
		elif (file_name[:4] == '5050') :
			content_value.append(reference_values[3])
		elif (file_name[:14] == 'McVitesCashews') :
			content_value.append(reference_values[4])
		elif (file_name[:4] == 'Moms') :
			content_value.append(reference_values[5])
		elif (file_name[:13] == 'GoodDayButter') :
			content_value.append(reference_values[6])
		elif (file_name[:11] == 'GoodDayNuts') :
			content_value.append(reference_values[7])
		elif (file_name[:13] == 'GoodDayCachew') :
			content_value.append(reference_values[8])
		elif (file_name[:16] == 'McVitesDigestive') :
			content_value.append(reference_values[9])

data = np.array(data)
content_value = np.array(content_value).reshape(len(data),1)
whole = np.hstack([content_value,data])

col_heads = ['value']
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

#Splitting the data into content values and raw data...
y = df[df.columns[0]]
X = df.drop('value',axis = 1)
y = pd.to_numeric(y)
X = X.astype(float)

#Training and testing data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 987, stratify = y)

#Pre-processing...
#scaler = preprocessing.StandardScaler().fit(X_train)								#The transformation to be used on training and testing sets
#X_train_scaled = scaler.transform(X_train)									#Standardising the training data (mean = 0,variance = 1) along each dimension
#X_test_scaled = scaler.transform(X_test)									#Standardising the testing data

pipeline = make_pipeline(preprocessing.StandardScaler(),GradientBoostingRegressor(random_state = 987))
#print pipeline.get_params()											#Listing out the hyperparameters that can be fine-tuned
hyperparameters = {'gradientboostingregressor__loss' : ['lad','ls','huber','quantile'],'gradientboostingregressor__learning_rate' : [1e-3,1e-1,1e-2,1e0,1e1],'gradientboostingregressor__max_depth' : [3,4,5,6,7,8,9],'gradientboostingregressor__max_features' : ['auto','sqrt','log2',None]}
clf = GridSearchCV(pipeline,hyperparameters,cv  = 4)
clf.fit(X_train.values,y_train.values)
print clf.best_params_												#To view the optimum values of the hyperparameters

pred = clf.predict(X_test)
print "R^2 score : ",r2_score(y_test,pred)											#Between -1 and 1, with -1 being bad model and 1 being accurate model
rmse = np.sqrt(mean_squared_error(y_test,pred))
print "RMSE : ",rmse
joblib.dump(clf, 'GradientBoost_regressor.pkl')			

#To load the model and test it on another test set
#clf2 = joblib.load('GradientBoost_regressor.pkl')
#clf2.predict(test_samples)

