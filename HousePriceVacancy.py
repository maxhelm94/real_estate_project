import pandas
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


######################################################
# Get 3 bed houses csv
housesDF = pd.read_csv('house3price.csv')

# Get taxable income data
vacancy = pd.read_csv('Vacancy.csv')

######################################################
# get the values of 3-bed houses in form of lists
postcodes = housesDF.iloc[:, 1].values.tolist()
jan12 = housesDF.iloc[:, 34].values.tolist()
jan22 = housesDF.iloc[:, 154].values.tolist()

# get the values of income data from jan12 to jan22
vacancyPostcodes = vacancy.iloc[:,1].values.tolist()
vacancyJan12 = vacancy.iloc[:,86].values.tolist()
vacancyJan22 = vacancy.iloc[:,206].values.tolist()
vacancyData = vacancy.iloc[:,86:206].values.tolist()
######################################################
# Clean property price data which has zero for values
del_list = []
for idx,x in enumerate(jan12):
    if x == '0':
        del_list.append(idx)

for idx,x in enumerate(jan22):
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

for idx,x in enumerate(vacancyData):
    for idy,y in enumerate(vacancyData[idx]):
        if str(vacancyData[idx][idy]) == '0.0':
            del_list.append(idx)

# for idx,x in enumerate(vacancyJan12):
#     if str(x) == '0.0':
#         del_list.append(idx)

# for idx,x in enumerate(vacancyJan22):
#     if str(x) == '0.0':
#         del_list.append(idx)

# sort the indices that should be erased; remove duplicates
del_list.sort()
del_list2 = [*set(del_list)]
del_list2.sort(reverse=True)

for x in del_list2:
    futureDeletePostcodes.append(str(vacancyPostcodes[x]))

######################################################
# Check for postcodes that don't exist in both lists
for idx,postcode in enumerate(postcodes):
    if postcode not in vacancyPostcodes:
        futureDeletePostcodes.append(str(postcode))

for idx,postcode in enumerate(vacancyPostcodes):
    if postcode not in postcodes:
        futureDeletePostcodes.append(str(postcode))

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
    del jan12[x]
    del jan22[x]
    del postcodes[x]

# Delete postcodes in taxable income data
finalDeleteList = []
for idx,x in enumerate(vacancyPostcodes):
    for y in futureDeletePostcodesSorted:
        if str(x) == str(y):
            finalDeleteList.append(idx)

# sort the indices that should be erased; remove duplicates
finalDeleteList.sort()
finalDeleteList2 = [*set(finalDeleteList)]
finalDeleteList2.sort(reverse=True)

for x in finalDeleteList2:
    # del vacancyJan12[x]
    # del vacancyJan22[x]
    del vacancyPostcodes[x]
    del vacancyData[x]

######################################################
# Clean property price data which has commas
comma = ','
for idx,stri in enumerate(jan12):
    if comma in stri:
        jan12[idx] = (int(stri.replace(',', '')))

for idx,stri in enumerate(jan22):
    if comma in stri:
        jan22[idx] = (int(stri.replace(',', '')))

######################################################
yData = []
for idx,x in enumerate(jan12):
    yData.append(float((float(jan22[idx]) / float(jan12[idx])) - 1 ) * 100)

# for idx,x in enumerate(vacancyJan12):
#     yData.append(x)

########################################################
# Format x-axis data for regression analysis
xData = []
for idx,x in enumerate(vacancyData):
    totalVacancy = 0
    for idy,y in enumerate(vacancyData[idx]):
        totalVacancy += vacancyData[idx][idy]
    xData.append(totalVacancy/len(vacancyData[0]))



xDataFormatted = []
for x in xData:
    tempList = []
    tempList.append(x)
    xDataFormatted.append(tempList)


vacancySeries = pandas.Series(index=vacancyPostcodes, data=xData)


########################################################
# Splitting the dataset into the Training set and Test set
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xDataFormatted, yData, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('House Price vs Vacancy (Training set)')
plt.xlabel('Vacancy')
plt.ylabel('House Price Growth (%)')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('House Price vs Vacancy')
plt.xlabel('Vacancy')
plt.ylabel('House Price Growth (%)')
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

####################################
X2 = sm.add_constant(xData)
est = sm.OLS(yData, X2)
est2 = est.fit()
print(est2.summary())
'''
