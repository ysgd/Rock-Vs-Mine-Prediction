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
print(sonar_data[60].value_counts())

print(sonar_data.groupby(60).mean())

#separating data and labels
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]
print(x)
print(y)
