#Importing Dependancies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#loading the dataset to a pandas DataFrame
sonar_data = pd.read_csv('dataset/Sonar Data.csv', header=None)
#print(sonar_data.head())

#print(sonar_data.shape) just to check number of rows and columns

#printing statistical values
print(sonar_data.describe())

#finding how many mines and rocks examples
""" print(sonar_data[60].value_counts())

print(sonar_data.groupby(60).mean()) """

#separating data and labels
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]
""" print(x)
print(y)
 """

#Training and Test Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, stratify = y, random_state=1)
#print(x.shape, x_train.shape, x_test.shape)

#Model Training -- using logistics regression
model = LogisticRegression()
model.fit(x_train, y_train)

#accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
#print('Accuracy Score: ', training_data_accuracy)

#accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
#print('Accuracy on test data: ', test_data_accuracy)

#Creating predictive system
input_data = (0.0210,0.0121,0.0203,0.1036,0.1675,0.0418,0.0723,0.0828,0.0494,0.0686,0.1125,0.1741,0.2710,0.3087,0.3575,0.4998,0.6011,0.6470,0.8067,0.9008,0.8906,0.9338,1.0000,0.9102,0.8496,0.7867,0.7688,0.7718,0.6268,0.4301,0.2077,0.1198,0.1660,0.2618,0.3862,0.3958,0.3248,0.2302,0.3250,0.4022,0.4344,0.4008,0.3370,0.2518,0.2101,0.1181,0.1150,0.0550,0.0293,0.0183,0.0104,0.0117,0.0101,0.0061,0.0031,0.0099,0.0080,0.0107,0.0161,0.0133)

#converting data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
    print("The object is a Rock")
else:
    print("The object is Mine")
