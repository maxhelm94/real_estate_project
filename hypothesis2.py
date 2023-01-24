import math
import statistics

import pandas
from hypothesis1 import model, hypothesis1, priceGrowthData3
from property_data_cleaning import postcodes

# list of postcodes in major urban areas namely
list_suburban_postcodes = []

# Sydney
x = 2000
while True:
    list_suburban_postcodes.append(x)
    if x == 2234:
        break
    x += 1

list_suburban_postcodes.append(2557)
list_suburban_postcodes.append(2558)
list_suburban_postcodes.append(2559)

# Brisbane
x = 4000
while True:
    list_suburban_postcodes.append(x)
    if x == 4184:
        break
    x += 1

y = [4205, 4207, 4300, 4301, 4303, 4304, 4305, 4306, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4508, 4509, 4510, 4512, 4516]
list_suburban_postcodes.extend(y)

# Melbourne
x = 3000
while True:
    list_suburban_postcodes.append(x)
    if x == 3207:
        break
    x += 1

y = [3335, 3752, 3754, 3765, 3766, 3767, 3781, 3782, 3785, 3786, 3787, 3788, 3789, 3791]
list_suburban_postcodes.extend(y)

# Perth
x = 6000
while True:
    list_suburban_postcodes.append(x)
    if x == 6175:
        break
    x += 1

# Adelaide
x = 5000
while True:
    list_suburban_postcodes.append(x)
    if x == 5174:
        break
    x += 1
list_suburban_postcodes.extend([5950, 5960])

# Canberra
list_suburban_postcodes.extend([2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2620])

# Newcastle
x = 2280
while True:
    list_suburban_postcodes.append(x)
    if x == 2308:
        break
    x += 1
list_suburban_postcodes.extend([2267, 2278])

# Gold Coast
x = 4207
while True:
    list_suburban_postcodes.append(x)
    if x == 4230:
        break
    x += 1

# removal is a list which lists all postcodes that are in list_suburban_postcodes but not in the model
removal = []
for i in list_suburban_postcodes:
    if i not in model.index:
        removal.append(i)
for i in removal:
    list_suburban_postcodes.remove(i)

cityDict = {}
for i in postcodes:
    if i in list_suburban_postcodes:
        cityDict[i] = 1
    else:
        cityDict[i] = 0

model = pandas.DataFrame({"past-future outperformance": hypothesis1, "city": cityDict})

model2 = model.assign(priceGrowth=priceGrowthData3)


def annualise(x):
    return x**(1/10) - 1

city_df = model2[model2['city'] == 1]
city_series = city_df['priceGrowth']
city_series = city_series.apply(annualise)

countryside_df = model2[model2['city'] == 0]
countryside_series = countryside_df['priceGrowth']
countryside_series = countryside_series.apply(annualise)

countrysideVariance = statistics.variance(countryside_series)
cityVariance = statistics.variance(city_series)

countryN = len(countryside_series)
cityN = len(city_series)
print(countryN)

countryMean = countryside_series.mean()
cityMean = city_series.mean()

print(f"city mean: {cityMean}\ncountry mean: {countryMean}")

stderr = math.sqrt(countrysideVariance/countryN + cityVariance/cityN)

zScore = (cityMean - countryMean)/stderr
print(zScore)

print(f"city median: {city_series.median()}\ncountryside median: {countryside_series.median()}")

