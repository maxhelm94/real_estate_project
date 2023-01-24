import pandas
import numpy
import matplotlib.pyplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


from hypothesis2 import model
from hypothesis1 import priceGrowthData3

def lists_to_numpy_array(x1, x2, x3):
    x = []
    i = 0
    for row in x1:
        x.append([x1[i], x2[i], x3[i]])
        i += 1

    x = numpy.array(x)

    return x


filehandle = pandas.read_csv(r'C:\Users\maxhe\PycharmProjects\COMM3303\COMM3303 Workplan - Distance from CBD.csv')

distance = filehandle.iloc[1:, 1].values.tolist()
postcodes = filehandle.iloc[1:, 0].values.tolist()

distance2 = []
for dis in distance:
    x = dis.split()
    distance2.append(x[0])

# distance2 = [float(x) for x in distance2]

distance_series = pandas.Series(index=postcodes, data=distance2)


duplicate_list = []
del_list = []
i = 0
for index, row in distance_series.items():
    if index in duplicate_list:
        del_list.append(i)
    else:
        duplicate_list.append(index)
    i += 1


distance1 = []
postcodes1 = []
j = 0

for i in postcodes:
    if j not in del_list:
        distance1.append(distance2[j])
        postcodes1.append(i)
    j += 1

distance_series = pandas.Series(index=postcodes1, data=distance1)

distance_series = distance_series[distance_series != 'undefined']

model1 = model.assign(distanceToCBD=distance_series)
model1 = model1[model1['distanceToCBD'].notna()]

distance_series1 = pandas.to_numeric(model1['distanceToCBD'])

model2 = model.assign(distanceToCBD=distance_series1)
model2 = model2.assign(priceGrowth=priceGrowthData3)
model2 = model2[model2['distanceToCBD'].notna()]
# print(model2)

duplicate_list = []
for index, row in model2.iterrows():
    if index in duplicate_list:
        pass
        # print(index)
    else:
        duplicate_list.append(index)


x = model2['distanceToCBD'].values

y = model2['priceGrowth'].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

x_train2 = []
for x in x_train:
    x_train2.append([x])

x_test2 = []
for x in x_test:
    x_test2.append([x])


regressor.fit(x_train2, y_train)

y_pred = regressor.predict(x_test2)

i = 0
for row in y_test:
    # print(f"{y_test[i]} {y_pred[i]}")
    i += 1
coefficient_of_dermination = r2_score(y_test, y_pred)
# print(coefficient_of_dermination)


matplotlib.pyplot.scatter(x_train2, y_train, color='red')
'''
matplotlib.pyplot.plot(x_train2, regressor.predict(x_test2), color='blue')

matplotlib.pyplot.title('crime/capital growth regression')
matplotlib.pyplot.xlabel('Crime Score')
matplotlib.pyplot.ylabel('Capital Growth')
'''
# matplotlib.pyplot.show()
