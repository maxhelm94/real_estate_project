# first import libraries

import numpy
import matplotlib.pyplot as plt
import pandas

# read in the file
dataset = pandas.read_csv('Salary_Data.csv')

# take all rows and all columns except the last (the last is
# for the dependent variable) 
x = dataset.iloc[:,:-1].values
# get just the last column for the dependent variable
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


from sklearn.linear_model import LinearRegression
# create an instance of the LinearRegression Class
regressor = LinearRegression()

# method that trains the regression model on the training set
regressor.fit(x_train, y_train)

# we want to create a predict function, which will output the dependent variable, when
# inputing the independent ones

# make the prediction by inputing the test set results
# will return a vector that indicates the predicted dependent variable for each set of independent inputs

# y_pred are the predicted salaries
y_pred = regressor.predict(x_test)

# visualize the training set results

# scatter will visualize the real data as red points on the graph
plt.scatter(x_train, y_train, color= 'red')

# now plot the regression line
print(x_train)
print(regressor.predict(x_train))
plt.plot(x_train, regressor.predict(x_train), color= 'blue')

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# show will display the graphic
plt.show()


# do the same for the test set
plt.scatter(x_test, y_test, color= 'red')
plt.plot(x_train, regressor.predict(x_train), color= 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()