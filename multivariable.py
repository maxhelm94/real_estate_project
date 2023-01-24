import matplotlib.pyplot
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from hypothesis3 import crimeModel1

x1 = crimeModel1['crime'].values
x2 = crimeModel1['city'].values
x3 = crimeModel1['past-future outperformance'].values

'''
    Use absolute values to compare the future/past outperformance 
'''
x3 = numpy.absolute(x3)

'''
    Develop a numpy array that contains all variable to work with
'''
x = []
i = 0
for row in x1:
    x.append([x1[i], x2[i], x3[i]])
    i += 1

x = numpy.array(x)

y = crimeModel1['priceGrowth'].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

i = 0
for row in y_test:
    print(f"{y_test[i]} {y_pred[i]}")
    i += 1
coefficient_of_dermination = r2_score(y_test, y_pred)

print(coefficient_of_dermination)

# numpy.set_printoptions(precision=2)
# print(numpy.concatenate((y_pred.reshape(len(y_pred), 1), (len(y_test), 1)), axis=1))

# matplotlib.pyplot.scatter(x_train, y_train, color='red')

# matplotlib.pyplot.plot(x_train, regressor.predict(x_test2), color='blue')
'''
matplotlib.pyplot.title('crime/capital growth regression')
matplotlib.pyplot.xlabel('Crime Score')
matplotlib.pyplot.ylabel('Capital Growth')
matplotlib.pyplot.show()

'''