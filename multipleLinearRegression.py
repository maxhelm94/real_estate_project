import pandas
import numpy
import matplotlib.pyplot as plt

dataset = pandas.read_csv('50_Startups.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# until here everything is always the same 

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# change the states into numerical values that are independent of one another (transform them to vectors)

# this index is 3 because we want to change the fourth column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = numpy.array(ct.fit_transform(x))
# print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(y_test)
# print(type(x_train[0]))
# we don't need to take the dummy variable trap in account because python does it for us

# similarly, python will find the best independent variables for our model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# is the vector of the predicted profit
y_pred = regressor.predict(x_test)

# precision = 2 means that numbers are only displayed to the second digit after comma
numpy.set_printoptions(precision=2)

# reshape will switch a vectors rows and columns
# will compare the real values to the predicted values
# print(numpy.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_pred), 1)), axis=1))