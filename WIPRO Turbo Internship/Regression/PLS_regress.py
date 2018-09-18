import numpy as np
import glob
import os
import pandas as pd

from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression,PLSSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale

reference_values = []
lines = 0
with open("/home/WIPRO Turbo Internship/Datasets/Biscuits/NutritionData.csv","r") as reference :
#with open("NutritionData.csv","r") as reference :
	for l in reference :
		lines += 1
		#2 - Calories, 3 - Total Fat, 4 - Saturated Fat, 5 - Poly Unsaturated Fat, 6 - Mono Unsaturated Fat, 7 - Cholestrol, 8 - Carbohydrates, 9 - Sugar, 10 - Protein 
		if (lines == 2) :
			currentline = l.split(",")
			for i in range(2,12) :
				reference_values.append(float(currentline[i]))

path = "/home/WIPRO Turbo Internship/Datasets/Biscuits/BiscuitsData/"

data,content_value = [],[]

for file_name in glob.glob(path + "*.csv") :
#	print file_name
	intensity = []
	lines = 0
	with open(file_name, "r") as filestream :
		for line in filestream :
			lines += 1
			if (lines > 20) :
				currentline = line.split(",")
			    	intensity.append(float(currentline[1]))
#			    	print currentline[1]
		data.append(intensity)
		file_name = os.path.basename(file_name)
#		print name		
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

#print np.shape(data),np.shape(content_value)

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
X_reduced = pca.fit_transform(scale(X))
n = len(X_reduced)
print n

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 987, stratify = y)

kf_10 = cross_validation.KFold(n,n_folds = 10,shuffle = True,random_state = 987)

#for train_index, test_index in kf_10 :
#	print train_index,test_index

mse = []
min_score = 1e20
for i in range(1,30) :
	pls = PLSRegression(n_components = i,scale = False)
	pls.fit(scale(X_reduced),y)
	score = -1 * cross_validation.cross_val_score(pls,X_reduced,y,cv = kf_10,scoring = 'neg_mean_squared_error').mean()
	if (score < min_score) :
		min_score = score
		min_id = i
	mse.append(score)

pls = PLSRegression(n_components = min_id)	

pls.fit(scale(X_train),y_train)
pred = pls.predict(scale(X_test))
rmse = np.sqrt(mean_squared_error(y_test,pred))
print "RMSE : ",rmse
print "R^2 score : ",r2_score(y_test,pred)
