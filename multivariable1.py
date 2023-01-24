import math
import statistics

import pandas
import numpy
from sklearn.metrics import r2_score



from hypothesis4 import lists_to_numpy_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from hypothesis2 import model
from hypothesis4 import model1
from hypothesis6 import df
from hypothesis5 import cheap_dataframe
from HousePriceVacancy import vacancySeries, yData, xData, xDataFormatted
from TaxableIncomeHousePriceGrowth import incomeGrowthSeries

model3 = model1.join(df)
model3 = model3.assign(vacancy=vacancySeries)
model3 = model3.assign(incomeGrowth=incomeGrowthSeries)
# model3['past-future outperformance'] = model3['past-future outperformance'].abs()


# model3 = model3[model3['State'] == '']
# model3 = model3[model3['Closest City'] == 'Gold Coast']
# model3 = model3[model3['city'] == 0]
model3 = model3.dropna()

# print(model3.columns)

y = model3['priceGrowth'].values

model5 = model3.drop(['Closest City', 'priceGrowth', 'distanceToCBD', 'State'], axis=1)

print(model3)

x = []
for index, row in model5.iterrows():
    lst = []
    for z in row:
        lst.append(z)
    x.append(lst)



y1 = []
for value in y:
    y1.append((value - 1) * 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

r2score = r2_score(y_test, y_pred)
print(f"r2 score: {r2score}")
# r2 = 9.8%
adj_r2 = 1-(1-r2score)*(len(y_test)-1)/(len(y_test)-5-1)
print(f"adj_r2 score: {adj_r2}")
# adj r2 = 7.5%

y4 = y_pred.tolist()

y2 = []
for i in y_test:
    y2.append((i**0.1 - 1)* 100)


y3 = []
for i in y4:
    y3.append((i**0.1 - 1)* 100)
i = 0
accept = 0
reject = 0
# print(y3)

import collections

discrepancy = []
dictionary = {}
for z in y3:
    # print(f'Prediction: {z}, Real Value: {y2[i]}')
    dictionary[z] = y2[i]
    discrepancy.append(abs(z - y2[i]))
    if (y2[i] - 0.5) < z < (y2[i] + 0.5):
        accept += 1
    else:
        reject += 1
    i += 1

#print(collections.OrderedDict(sorted(dictionary.items())))

print(f'Average discrapency: {sum(discrepancy)/len(discrepancy)}')
print(f'Accepted: {accept}, Rejected: {reject}, Accept rate: {accept/(accept + reject)}')

difference = []
difference1 = []
for key, value in dictionary.items():
    interim = value - key
    #print(interim)
    difference1.append(interim)

    if -1 < interim < 1:
        interim = 0
    elif interim < -1:
        interim += 1
    elif interim > 1:
        interim -= 1

    difference.append(interim)

print(f"sum of difference: {sum(difference)}\nsum of difference1: {sum(difference1)}")

variance_difference = statistics.variance(difference1)
variance_predictions = statistics.variance(y3)
variance_realValues = statistics.variance(y2)
variance_total = statistics.variance(y2 + y3)

mean_difference = sum(difference)/len(difference)
mean_predictions = sum(y3)/len(y3)
mean_realValues = sum(y2)/len(y2)
mean_total = sum(y2 + y3)/(len(y2) + len(y3))

n_difference = len(difference)
n_predictions = len(y3)
n_realValues = len(y2)
n_total = len(y2 + y3)


# stdrr = math.sqrt(variance_predictions/n_predictions + variance_realValues/n_realValues)
stdrr = math.sqrt(variance_difference/n_difference)
z_score = (mean_difference)/stdrr
print(f"Variance: {variance_difference}\nMean: {mean_difference}\nSample size: {n_difference}\nstdrr: {stdrr}")
print(z_score)



'''
import statistics
# mean = sum(y1)/len(y1)
# variance = statistics.variance(cheap_dataframe['priceGrowth'].values)
# print(f"Overall mean: {mean} Overall variance: {variance}")

cheapDataframe = model3[model3['incomeGrowth'] >= 6500]
expensiveDataframe = model3[model3['incomeGrowth'] < 6500]

meancheap = sum(cheapDataframe['priceGrowth'].values)/len(cheapDataframe['priceGrowth'])
meanexpensive = sum(expensiveDataframe['priceGrowth'].values)/len(expensiveDataframe['priceGrowth'])

cheapvariance = statistics.variance(cheapDataframe['priceGrowth'].values)
expensivevariance = statistics.variance(expensiveDataframe['priceGrowth'].values)

print(f"cheap dataset mean: {meancheap} expensive dataset mean: {meanexpensive}")
print(f"cheap dataset variance: {cheapvariance} expensive dataset variance: {expensivevariance}")

cheapN = len(cheapDataframe['priceGrowth'])
expensiveN = len(expensiveDataframe['priceGrowth'])

print(f"cheap dataset size: {cheapN} expensive dataset size: {expensiveN}")

stderr = math.sqrt(cheapvariance/cheapN + expensivevariance/expensiveN)
# stderr = math.sqrt(cheapvariance/cheapN + variance/len(cheap_dataframe["priceGrowth"].values))

print(f"standard error: {stderr}")

pooledVariance = statistics.variance(model3['priceGrowth'])

interval_start = (meancheap - meanexpensive) - 1.96 * stderr
interval_stop = (meancheap - meanexpensive) + 1.96 * stderr
print(f"interval is: {interval_start} - {interval_stop}")

population_mean = cheapDataframe['priceGrowth'].mean()
sample_mean = population_mean + 0.1046

z_score = (meancheap - meanexpensive)/stderr
print(f"population mean: {population_mean} z_score: {z_score}")
# z = 0

'''