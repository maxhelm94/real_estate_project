import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm


from multivariable1 import model3

# model4 = model3[model3['city'] == 1]
# model4 = model4[model4['Closest City'].isin(['Brisbane']) == 'Brisbane']
model4 = model3[model3['State'].isin(['Vic'])]
model4 = model4[model4['vacancy'] < 11]

yData = model4['priceGrowth'].values

xData = model4['vacancy'].values

xDataFormatted = []
for x in xData:
    tempList = []
    tempList.append(x)
    xDataFormatted.append(tempList)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xDataFormatted, yData, test_size = 1/3, random_state = 3)

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
plt.title('House Price vs Vacancy in Qld and Nsw')
plt.xlabel('Vacancy')
plt.ylabel('House Price Growth (%)')
plt.show()

from sklearn.metrics import r2_score
print(f"r2 score vacancy: {r2_score(y_test, y_pred)}")

####################################
X2 = sm.add_constant(xData)
est = sm.OLS(yData, X2)
est2 = est.fit()
print(est2.summary())