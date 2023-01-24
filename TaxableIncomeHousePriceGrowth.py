import pandas
import pandas as pd
import matplotlib.pyplot as plt

######################################################
# Get 3 bed houses csv
housesDF = pd.read_csv('house3price.csv')

# Get taxable income data
taxableIncome = pd.read_csv('TaxableIncomeData.csv')

######################################################
# get the values of 3-bed houses in form of lists
postcodes = housesDF.iloc[:, 1].values.tolist()
dec13 = housesDF.iloc[:, 57+12].values.tolist()
dec19 = housesDF.iloc[:, 129+12].values.tolist()

# get the values of income data from dec13 to dec19
postcodeTaxableIncome = taxableIncome.iloc[:,1].values.tolist()
medianTaxableIncome13To14 = taxableIncome.iloc[:,6].values.tolist()
medianTaxableIncome19To20 = taxableIncome.iloc[:,24].values.tolist()

######################################################
# Clean property price data which has zero for values
del_list = []
for idx,x in enumerate(dec13):
    if x == '0':
        del_list.append(idx)

for idx,x in enumerate(dec19):
    if x == '0':
        del_list.append(idx)

# sort the indices that should be erased; remove duplicates
del_list.sort()
del_list2 = [*set(del_list)]
del_list2.sort(reverse=True)

futureDeletePostcodes = []

for x in del_list2:
    futureDeletePostcodes.append(str(postcodes[x]))

# save all indices that have a 'na' as value
del_list = []

for idx,x in enumerate(medianTaxableIncome13To14):
    if x == 'na':
        del_list.append(idx)


for idx,x in enumerate(medianTaxableIncome19To20):
    if x == 'na':
        del_list.append(idx)


# sort the indices that should be erased; remove duplicates
del_list.sort()
del_list2 = [*set(del_list)]
del_list2.sort(reverse=True)

for x in del_list2:
    futureDeletePostcodes.append(str(postcodeTaxableIncome[x]))

######################################################
# Check for postcodes that don't exist in both lists
for idx,postcode in enumerate(postcodes):
    if postcode not in postcodeTaxableIncome:
        futureDeletePostcodes.append(str(postcode))

for idx,postcode in enumerate(postcodeTaxableIncome):
    if postcode not in postcodes:
        futureDeletePostcodes.append(str(postcode))

######################################################
# Delete outlier postcode 6710
futureDeletePostcodes.append("6710")

######################################################
testList = []
for x in postcodes:
    if x in testList:
        futureDeletePostcodes.append(str(x))
    else:
        testList.append(x)

# sort the indices that should be erased; remove duplicates
futureDeletePostcodesSorted = sorted(set(futureDeletePostcodes))

# Delete postcodes in property price data
finalDeleteList = []
for idx,x in enumerate(postcodes):
    for y in futureDeletePostcodesSorted:
        if str(x) == str(y):
            finalDeleteList.append(idx)

# sort the indices that should be erased; remove duplicates
finalDeleteList.sort()
finalDeleteList2 = [*set(finalDeleteList)]
finalDeleteList2.sort(reverse=True)

for x in finalDeleteList2:
    del dec13[x]
    del dec19[x]
    del postcodes[x]

# Delete postcodes in taxable income data
finalDeleteList = []
for idx,x in enumerate(postcodeTaxableIncome):
    for y in futureDeletePostcodesSorted:
        if str(x) == str(y):
            finalDeleteList.append(idx)

# sort the indices that should be erased; remove duplicates
finalDeleteList.sort()
finalDeleteList2 = [*set(finalDeleteList)]
finalDeleteList2.sort(reverse=True)

for x in finalDeleteList2:
    del medianTaxableIncome13To14[x]
    del medianTaxableIncome19To20[x]
    del postcodeTaxableIncome[x]

######################################################
# Clean property price data which has commas
comma = ','
for idx,stri in enumerate(dec13):
    if comma in stri:
        dec13[idx] = (int(stri.replace(',', '')))

for idx,stri in enumerate(dec19):
    if comma in stri:
        dec19[idx] = (int(stri.replace(',', '')))

######################################################
# Calculate property percentage price growth
priceGrowth = []
for x in range(len(dec13)):
    percentage = (float((float(dec19[x]) / float(dec13[x])) - 1 ) * 100)
    priceGrowth.append(percentage)

######################################################
medianTaxableIncome13To14mod = []
comma = ','
for stri in medianTaxableIncome13To14:
    if comma in stri:
        medianTaxableIncome13To14mod.append(int(stri.replace(',', '')))

medianTaxableIncome19To20mod = []
for stri in medianTaxableIncome19To20:
    if comma in stri:
        medianTaxableIncome19To20mod.append(int(stri.replace(',', '')))

incomeDataFrom13To20 = []
for idx,x in enumerate(medianTaxableIncome13To14mod):
    incomeDataFrom13To20.append(int(medianTaxableIncome19To20mod[idx] - int(medianTaxableIncome13To14mod[idx])))

########################################################
# Format x-axis data for regression analysis
formattedIncomeDataFrom13To20 = []
for x in incomeDataFrom13To20:
    tempList = []
    tempList.append(x)
    formattedIncomeDataFrom13To20.append(tempList)

incomeGrowthSeries = pandas.Series(index=postcodeTaxableIncome, data=incomeDataFrom13To20)
'''
'''
    # List all suburbs that have a median income growth of 4500 to 5500
'''

incomeGrowth5000 = []
priceGrowth5000 = []
postcodeTaxableIncome5000 = []
i = 0


for x in incomeDataFrom13To20:
    if 4500 <= x <= 5500:
        incomeGrowth5000.append(incomeDataFrom13To20[i])
        priceGrowth5000.append(priceGrowth[i])
        postcodeTaxableIncome5000.append(postcodeTaxableIncome[i])
    i += 1

priceGrowth5000Frame = pandas.DataFrame(index=postcodeTaxableIncome5000, data={"priceGrowth": priceGrowth5000, "incomeGrowth": incomeGrowth5000})
priceGrowth5000Frame = priceGrowth5000Frame.sort_values(by=["priceGrowth"])

worstPerformers = priceGrowth5000Frame.iloc[0:9, :]
bestPerformers = priceGrowth5000Frame.iloc[-10:, :]
print(priceGrowth5000Frame)
print(worstPerformers)
print(bestPerformers)
'''
########################################################
# Splitting the dataset into the Training set and Test set
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(formattedIncomeDataFrom13To20, priceGrowth, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Median Income Growth 2013-19 vs House Price Growth 2015-21 (Training set)')
plt.xlabel('Median Income Growth')
plt.ylabel('House Price Growth')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Median Income Growth 2013-19 vs House Price Growth 2015-21')
plt.xlabel('Median Income Growth ($)')
plt.ylabel('House Price Growth (%)')
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
'''
