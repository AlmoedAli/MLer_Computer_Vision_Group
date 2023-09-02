import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib

# read file and get data
file= pd.read_csv('dataset.csv')
X= file.iloc[:, 0: 63].values
y= file.iloc[:, 63:].T.values
y= np.ravel(y)

#split trainSet and testSet
X_train_val, X_test, y_train_val, y_test= train_test_split(X, y, test_size= 0.2)

#Tuning hyparameters
para_Grid= {'n_estimators': [20, 30, 40],
            'min_samples_split': [2, 3, 5], 'max_depth': [5, 7, 10]}

grid= GridSearchCV(RandomForestClassifier(), para_Grid, refit= True)

grid.fit(X_train_val, y_train_val)

model= grid.best_estimator_

# print(model.get_params)
# print(type(X))
# print(X[0])
# print(model.predict(X_test[0]))
# y_predict= model.predict(X_test)
# print(np.sum(y_predict== y_test)/ y_test.shape[0])
joblib.dump(model, 'model.joblib')
