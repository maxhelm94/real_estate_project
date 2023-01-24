# if the data set is too small, we won't split it in test and
# train set but use the whole set to train it

import pandas
import numpy
import matplotlib.pyplot as plt

dataset = pandas.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)


from sklearn.preprocessing import PolynomialFeatures
# the 2 in the brackets is the exponent of our variable 
poly_reg = PolynomialFeatures(degree=4)
# transform the data of the independent variable
x_poly = poly_reg.fit_transform(x)

# create a linear regression model with the polynomial independent variable as
# the input values
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)


# we first want to plot the correct coordinates with the real values
# plt.scatter(x, y, color='red')
# plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# plt.show()

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# if we want to check a certain value, we have to input 6.5 into 2 square brackets
# as it only takes a 2D array as input
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))