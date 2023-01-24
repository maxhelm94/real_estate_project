import statistics
import pandas
import matplotlib.pyplot

from hypothesis2 import model
from hypothesis1 import priceGrowthData3
from sklearn.metrics import r2_score


# Criminality data
filehandle = pandas.read_csv("CrimeDataNSW95-21.csv")
postcodes_crime = filehandle.iloc[1:, 0]
offense = filehandle.iloc[1:, 207:]
offenseName = filehandle.iloc[1:, 1]

crime_model = offense.assign(postcodes=postcodes_crime)
crime_model = crime_model.assign(crimetype=offenseName)
# print(crime_model)

crimeScore = {}
for index, row in crime_model.iterrows():
    if row.postcodes in model.index:
        score = 0
        if row.crimetype == "Homicide":
            for homicide in row[1:-2]:
                x = homicide * 9
                score += x
        if row.crimetype == "Assault":
            for homicide in row[1:-2]:
                x = homicide * 3
                score += x
        if row.crimetype == "Sexual offences":
            for homicide in row[1:-2]:
                x = homicide * 7
                score += x
        if row.crimetype == "Abduction and kidnapping":
            for homicide in row[1:-2]:
                x = homicide * 8
                score += x
        if row.crimetype == "Robbery":
            for homicide in row[1:-2]:
                x = homicide * 7
                score += x
        if row.crimetype == "Blackmail and extortion":
            for homicide in row[1:-2]:
                x = homicide * 3
                score += x
        if row.crimetype == "Intimidation, stalking and harassment":
            for homicide in row[1:-2]:
                x = homicide * 3
                score += x
        if row.crimetype == "Other offences against the person":
            for homicide in row[1:-2]:
                x = homicide * 1
                score += x
        if row.crimetype == "Theft":
            for homicide in row[1:-2]:
                x = homicide * 1
                score += x
        if row.crimetype == "Arson":
            for homicide in row[1:-2]:
                x = homicide * 2
                score += x
        if row.crimetype == "Malicious damage to property":
            for homicide in row[1:-2]:
                x = homicide * 2
                score += x
        if row.crimetype == "Drug offences":
            for homicide in row[1:-2]:
                x = homicide * 2
                score += x
        if row.crimetype == "Prohibited and regulated weapons offences":
            for homicide in row[1:-2]:
                x = homicide * 2
                score += x
        if row.crimetype == "Disorderly conduct":
            for homicide in row[1:-2]:
                x = homicide * 1
                score += x
        if row.crimetype == "Betting and gaming offences":
            for homicide in row[1:-2]:
                x = homicide * 1
                score += x
        if row.crimetype == "Liquor offences":
            for homicide in row[1:-2]:
                x = homicide * 1
                score += x
        if row.crimetype == "Pornography offences":
            for homicide in row[1:-2]:
                x = homicide * 2
                score += x
        if row.crimetype == "Prostitution offences":
            for homicide in row[1:-2]:
                x = homicide * 2
                score += x
        if row.crimetype == "Against justice procedures":
            for homicide in row[1:-2]:
                x = homicide * 1
                score += x
        if row.crimetype == "Transport regulatory offences":
            for homicide in row[1:-2]:
                x = homicide * 1
                score += x
        if row.crimetype == "Other offences":
            for homicide in row[1:-2]:
                x = homicide * 1
                score += x

        if row.postcodes in crimeScore.keys():
            crimeScore[row.postcodes] += score
        else:
            crimeScore[row.postcodes] = score

crimeSeries = pandas.Series(crimeScore)
model = model.assign(crime=crimeSeries)
crimeModel = model[model['crime'].notna()]

crimeModel1 = crimeModel[crimeModel['city'] == 1]
# print(crimeModel1)
'''
    Calculate the CAGR over 10 years
'''

crimeModel1 = crimeModel1.assign(priceGrowth=priceGrowthData3)
crimeModel1.sort_index(inplace=True)

# crimeModel2 = crimeModel1[crimeModel1['crime'] > 80000]
# print(crimeModel2)

x = crimeModel1['crime'].values
y = crimeModel1['priceGrowth'].values


y1 = []
for value in y:
    y1.append((value - 1) * 100)

# print(crimeModel1)
from sklearn.model_selection import train_test_split

y_train, y_test, x_train, x_test = train_test_split(y1, x, test_size = 0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

y_train2 = []
for x in y_train:
    y_train2.append([x])

y_test2 = []
for x in y_test:
    y_test2.append([x])


regressor.fit(y_train2, x_train)

y_pred = regressor.predict(y_test2)

matplotlib.pyplot.scatter(y_train2, x_train, color='red')
matplotlib.pyplot.plot(y_train2, regressor.predict(y_train2), color='blue')

matplotlib.pyplot.title('Crime rate and capital growth are not correlated')
matplotlib.pyplot.xlabel('Capital Growth in %')
matplotlib.pyplot.ylabel('Crime Score')
matplotlib.pyplot.show()

low_crime_cap_growth = crimeModel1[crimeModel1['crime'] <= 50000]
stddev_low_crime = statistics.stdev(low_crime_cap_growth['priceGrowth'].values)
# print(stddev_low_crime)

high_crime_cap_growth = crimeModel1[crimeModel1['crime'] > 50000]
stddev_high_crime = statistics.stdev(high_crime_cap_growth['priceGrowth'].values)
# print(stddev_high_crime)

coefficient_of_dermination = r2_score(x_test, y_pred)
print(coefficient_of_dermination)
# r2 = 0.6%
