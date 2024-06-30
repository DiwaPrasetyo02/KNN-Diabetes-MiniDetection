# -*- coding: utf-8 -*-
"""KNN.ipynb
"""

import pandas as pd

data = pd.read_csv('diabetes.csv', sep=',')

data.head()

display(data)

data.info()

Missing_Values = data.isnull().sum()
print("Missing value = \n" ,Missing_Values)

Duplicate_value = data.duplicated().sum()
print ("Duplicate Value : ", Duplicate_value)

data = data.drop_duplicates()
print ("duplicate value sesudah before : ", data.duplicated().sum())

data.shape

'''
1 = kena
0 = tidak kena
import time
'''

data_x = data.drop('Outcome',axis=1)
display(data_x)
data_y = data['Outcome']
display(data_y)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

#start = time.time()
#end = time.time()
#print(f"Model training completed in {end-start}")

#model evalutation
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)

confussion_result = confusion_matrix(y_test, y_pred)
print(confussion_result)

new_data = pd.DataFrame({
    'Pregnancies' : [0],
    'Glucose' : [330],
    'BloodPressure' : [150],
    'SkinThickness' :  [25],
    'Insulin' : [1],
    'BMI' : [38.5],
    'DiabetesPedigreeFunction' : [0.351],
    'Age' : [30],
})
display(new_data)


new_predict = model.predict(new_data)
display(new_predict)
