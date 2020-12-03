# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:14:58 2020

@author: shahe
"""

#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


#Import the data
cars = pd.read_csv("used_cars.csv")

#Check the head
cars.head()

#Check the shape
cars.shape

#Check the structure of dataset
cars.info()

#Find out the unique items in each of the categorical variable

for item in cars.columns:
    if cars[item].dtypes == 'object':
        print("Number of unique items in {}:- {}".format(item,cars[item].unique()))
        
#Check for any missing value
cars.isnull().sum()


cars_df = cars[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]

#Add number of years since the vehicle was purchased
cars_df['Year_diff'] = 2020 - cars_df['Year']

#Drop Year from dataset
cars_df.drop('Year',axis=1,inplace=True)

#One hot encode the categorical variables
cars_df = pd.get_dummies(cars_df,drop_first=True)


#Get the correlation of each variable
cars_corr = cars_df.corr()

#Plot Heatmap to visualize the correlation
plt.figure(figsize=(16,10))
sns.heatmap(cars_corr,annot=True,cmap='magma_r')

X = cars_df.iloc[:,1:]
y = cars_df.iloc[:,0]

#Feature Selection
from sklearn.ensemble import ExtraTreesRegressor

ftree = ExtraTreesRegressor()
ftree.fit(X,y)

feat_importances = pd.Series(ftree.feature_importances_,index = X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


#Do Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=99)


#Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


rf_model = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf_model,
                               param_distributions = random_grid,
                               scoring='neg_mean_squared_error', 
                               n_iter = 20, 
                               cv = 5, 
                               verbose=3,
                               random_state=99, 
                               n_jobs = -1)

rf_random.fit(x_train,y_train)

#Best parameter
rf_random.best_params_

#Best Score
rf_random.best_score_

#Make Predictions
y_pred = rf_random.predict(x_test)

sns.distplot(y_test-y_pred)
plt.scatter(y_test,y_pred)


from sklearn import metrics

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#Save the model
file = open('rf_reg_model.pkl','wb')
pickle.dump(rf_random,file)







