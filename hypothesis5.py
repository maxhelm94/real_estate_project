import math

import numpy
import pandas
import matplotlib.pyplot

from property_data_cleaning import jan12mod, postcodes
from hypothesis1 import priceGrowthData3
from hypothesis2 import model
from hypothesis6 import df
from sklearn.metrics import r2_score

cheap = pandas.Series(index=postcodes, data=jan12mod)


cheap_dataframe = model.assign(initial_investment=cheap)
# cheap_dataframe = cheap_dataframe.assign(priceGrowth=priceGrowthData3)
# pandas.DataFrame({"price": cheap, "priceGrowth": priceGrowthData3})
# cheap_dataframe = [cheap_dataframe['city']]
# print(cheap_dataframe)

cheap_dataframe = cheap_dataframe.join(df)
# cheap_dataframe = cheap_dataframe[cheap_dataframe['city'] == 1]
# cheap_dataframe = cheap_dataframe[cheap_dataframe['Closest City'].isin(['Sydney'])]
# cheap_dataframe = cheap_dataframe[cheap_dataframe['pri']]


cheap_dataframe = cheap_dataframe[cheap_dataframe['city'] == 1]
cheap_dataframe = cheap_dataframe[cheap_dataframe['priceGrowth'] < 4]
# print(cheap_dataframe.columns)
x = cheap_dataframe['initial_investment'].values
y = cheap_dataframe['priceGrowth'].values

x1 = []
for i in x:
    x1.append(i/1000)
x = x1

y1 = []
for i in y:
    y1.append((i - 1) * 100)
    y = y1


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size = 0.2, random_state=4)

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

matplotlib.pyplot.scatter(x_train2, y_train, color='red')
matplotlib.pyplot.plot(x_train2, regressor.predict(x_train2), color='blue')
# matplotlib.pyplot.plot([1000000, 2000000], [1, 2, 3], color='blue')

matplotlib.pyplot.title('cheaper suburbs do not outperform more expensive suburbs in cities')
matplotlib.pyplot.xlabel('Initial investment in 1000s')
matplotlib.pyplot.ylabel('Capital Growth in %')
matplotlib.pyplot.show()

coefficient_of_dermination = r2_score(y_test, y_pred)
print(coefficient_of_dermination)


############################################################################
'''
import statistics
mean = sum(y1)/len(y1)
variance = statistics.variance(cheap_dataframe['priceGrowth'].values)
print(f"Overall mean: {mean}\nOverall variance: {variance}")

cheapDataframe = cheap_dataframe[cheap_dataframe['initial_investment'] < 730000]
expensiveDataframe = cheap_dataframe[cheap_dataframe['initial_investment'] >= 730000]

meancheap = sum(cheapDataframe['priceGrowth'].values)/len(cheapDataframe['priceGrowth'])
meanexpensive = sum(expensiveDataframe['priceGrowth'].values)/len(expensiveDataframe['priceGrowth'])

cheapvariance = statistics.variance(cheapDataframe['priceGrowth'].values)
expensivevariance = statistics.variance(expensiveDataframe['priceGrowth'].values)

print(f"cheap dataset mean: {meancheap}\nexpensive dataset mean: {meanexpensive}")
print(f"cheap dataset variance: {cheapvariance}\nexpensive dataset variance: {expensivevariance}")

cheapN = len(cheapDataframe['priceGrowth'])
expensiveN = len(expensiveDataframe['priceGrowth'])

print(f"cheap dataset size: {cheapN}\nexpensive dataset size: {expensiveN}")

stderr = math.sqrt(cheapvariance/cheapN + expensivevariance/expensiveN)
# stderr = math.sqrt(cheapvariance/cheapN + variance/len(cheap_dataframe["priceGrowth"].values))

print(f"standard error: {stderr}")

pooledVariance = statistics.variance(cheap_dataframe['priceGrowth'])

interval_start = (meancheap - meanexpensive) - 1.96 * stderr
interval_stop = (meancheap - meanexpensive) + 1.96 * stderr
print(f"interval is: {interval_start} - {interval_stop}")

population_mean = cheapDataframe['priceGrowth'].mean()
# sample_mean = population_mean + 0.1046

z_score = ((meancheap - meanexpensive) - (0.00))/stderr
print(f"z_score: {z_score}")

'''
