import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

file= pd.read_csv('dataset.csv')

# file= loadtxt('dataset.csv', delimiter=',')
X= file.iloc[:, 0: 63].values
y= file.iloc[:, 63:].T.values
y= np.ravel(y)
# print(y)
X_train_val, X_test, y_train_val, y_test= train_test_split(X, y, test_size= 0.2)
X_train, X_val, y_train, y_val= train_test_split(X_train_val, y_train_val, test_size= 0.2)


model= RandomForestClassifier()
model.fit(X_train_val, y_train_val)

# print(type(X))
# print(X[0])
# print(model.predict(X_test[0]))
# y_predict= model.predict(X_test)
# print(np.sum(y_predict== y_test)/ y_test.shape[0])
joblib.dump(model, 'model.joblib')
