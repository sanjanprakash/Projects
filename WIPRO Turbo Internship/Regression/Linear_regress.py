import numpy as np
import glob
import os
import pandas as pd

from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale

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

pca = PCA()
X_reduced = pca.fit_transform(scale(X))										#Performing PCA transformation over re-scaled data
n = len(X_reduced)

kf = cross_validation.KFold(n,n_folds = 10, shuffle = True, random_state = 987)

regress = LinearRegression()

mse = []
score = -1 * cross_validation.cross_val_score(regress,np.ones((n,1)),y.ravel(),cv = kf, scoring = 'neg_mean_squared_error').mean()
mse.append(score)
min_score = 1e20

#Calculating the MSE using cross-validation after adding up each of the possible components one-by-one and finding the minimum MSE value
for i in range(1,n) :
	score = -1 * cross_validation.cross_val_score(regress,X_reduced[:,:i],y.ravel(),cv = kf, scoring = 'neg_mean_squared_error').mean()
	if (score < min_score) :
		min_id = i
	mse.append(score)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 987, stratify = y)

#PCA transformation on the training data...`
pca2 = PCA()
X_reduced_train = pca2.fit_transform(scale(X_train))
n = len(X_train)

kf_10 = cross_validation.KFold(n,n_folds = 10,shuffle = True,random_state = 987)

#Calculating the MSE using cross-validation after adding up each of the possible components one-by-one till 'min_id' components and finding the minimum MSE value
mse = []
score = -1 * cross_validation.cross_val_score(regress,np.ones((n,1)),y_train.ravel(),cv = kf_10,scoring = 'neg_mean_squared_error').mean()
mse.append(score)
min_score = 1e20
for i in range(1,min_id + 1) :
	score = -1 * cross_validation.cross_val_score(regress,X_reduced_train[:,:i],y_train.ravel(),cv = kf_10,scoring = 'neg_mean_squared_error').mean()
	if (score < min_score) :
		min_id2 = i
		min_score = score
	mse.append(score)

#PCA transformation over the test data	
X_reduced_test = pca2.transform(scale(X_test))[:,:min_id2]
regress = LinearRegression()
regress.fit(X_reduced_train[:,:min_id2],y_train)								#Fitting a linear regression model over the training data

pred = regress.predict(X_reduced_test)
rmse = np.sqrt(mean_squared_error(y_test,pred))
print "RMSE : ",rmse
print "R^2 score : ",r2_score(y_test,pred)
