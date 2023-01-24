import pandas

from property_data_cleaning import priceGrowth, postcodes, priceGrowth2, priceGrowth3, lst

# pandas Series that has price growth data for suburbs from 2012 to 2017 with the postcodes as index
priceGrowthData = pandas.Series(data=priceGrowth, index=postcodes)

# sort the aforementioned Series by its values (price growth) to ultimately create a Series that ranks
# the suburbs according to its price growth data
pGD = priceGrowthData.sort_values()
pGD = pandas.Series(data=lst, index=pGD.index.tolist())

# pandas Series that has price growth data for suburbs from 2017 to 2022 with the postcodes as index
priceGrowthData2 = pandas.Series(data=priceGrowth2, index=postcodes)
pGD2 = priceGrowthData2.sort_values()
pGD2 = pandas.Series(data=lst, index=pGD2.index.tolist())


# pandas Series that has price growth data for suburbs from 2012 to 2022 with the postcodes as index
priceGrowthData3 = pandas.Series(data=priceGrowth3, index=postcodes)
pGD3 = priceGrowthData3.sort_values()
pGD3 = pandas.Series(data=lst, index=pGD3.index.tolist())


# subtract the rank of past outperformance from current outperformance and save it in a new Series
hypothesis1 = pGD2.sub(pGD)
hypothesis1.sort_index()


model = pandas.DataFrame({"past-future outperformance": hypothesis1})

'''
    percentile for best performers in first period
'''
firstPercentile = 0.25
top_firstPercentile = int(firstPercentile * priceGrowthData.size)

'''
    flip the ascending flag in priceGrowthData to count from bottom up
'''
priceGrowthData = priceGrowthData.sort_values(ascending=True)
priceGrowthData2 = priceGrowthData2.sort_values(ascending=False)

'''
    percentile for best performers in the second period
'''
secondPercentile = 0.5
top_secondPercentile = int(secondPercentile * priceGrowthData.size)

# list of best firstPercentile performing suburbs from 2012 to 2017
x = 0
top_1_postcodes = []
for key, value in priceGrowthData.items():
    if x == top_firstPercentile:
        break
    top_1_postcodes.append(key)
    x += 1

# list of best secondPercentile performing suburbs from 2017 to 2022
x = 0
top_2_postcodes = []
for key, value in priceGrowthData2.items():
    if x == top_secondPercentile:
        break
    top_2_postcodes.append(key)
    x += 1

# check how many of the first list are in the second list
z = 0
x = 0
for postcode in top_1_postcodes:
    if postcode in top_2_postcodes:
        z += 1

average = z / len(top_1_postcodes)
# print(average)
# print(f"{z} of the postcodes are within the top {firstPercentile*100}th percentile of both periods")

