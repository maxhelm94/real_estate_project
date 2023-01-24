import pandas

from hypothesis1 import priceGrowthData3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot
from sklearn.metrics import r2_score


df = pandas.read_csv('DistanceToCBD.csv')

df = df[['Post Code', 'Closest City', 'Distance (number)', 'State']]

# df = df[df['Distance (number)'] < 70]
# df = df[df['Distance (number)'] > 30]


# df = df[df['Closest City'].isin(['Brisbane', 'Sydney', 'Adelaide', 'Perth', 'Newcastle', 'Canberra', 'Gold Coast', 'Melbourne'])]
# df = df[df['Distance (number)'] < 70]


df = df.set_index(['Post Code'])

df = df[~df.index.duplicated(keep='first')]


df = df.assign(priceGrowth=priceGrowthData3)

df = df[df['Distance (number)'] != 'undefined']
df = df.dropna()


x = df['Distance (number)'].values
y = df['priceGrowth'].values

'''
y1 = []
for value in y:
    y1.append((value - 1) * 100)


x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size = 0.2, random_state=0)
regressor = LinearRegression()

x_train2 = []
for x in x_train:
    x_train2.append([x])

x_test2 = []
for x in x_test:
    x_test2.append([x])


regressor.fit(x_train2, y_train)

y_pred = regressor.predict(x_test2)

matplotlib.pyplot.scatter(x_train2, y_train, color='red')
matplotlib.pyplot.plot(x_train2, regressor.predict(x_train2), color='blue')

matplotlib.pyplot.title('CBD closeness')
matplotlib.pyplot.ylabel('Capital Growth in %')
matplotlib.pyplot.xlabel('Distance to CBD in km')
# matplotlib.pyplot.show()

coefficient_of_dermination = r2_score(y_test, y_pred)
print(coefficient_of_dermination)
'''