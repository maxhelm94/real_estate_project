import pandas
import matplotlib.pyplot
import numpy

# test first hypothesis; past outperformance leads to future outperformance

dataset = pandas.read_csv('house3price.csv')

# get the values of 3-bed houses in form of lists
postcodes = dataset.iloc[:, 1].values.tolist()
jan12 = dataset.iloc[:, 34].values.tolist()
jan17 = dataset.iloc[:, 94].values.tolist()
jan22 = dataset.iloc[:, 154].values.tolist()


# save all indices that have a '0' as value in one of the 3 columns
del_list = []
y = 0
for x in jan12:
    if x == '0':
        del_list.append(y)
    y += 1

y = 0
for x in jan17:
    if x == '0':
        del_list.append(y)
    y += 1

y = 0
for x in jan22:
    if x == '0':
        del_list.append(y)
    y += 1

y = 0
del_list2 = []
for x in postcodes:
    if x in del_list2:
        del_list.append(y)
    else:
        del_list2.append(x)
    y += 1

# sort the indices that should be erased; remove duplicates
del_list.sort()
del_list2 = [*set(del_list)]
del_list2.sort(reverse=True)


for x in del_list2:
    del jan12[x]
    del jan17[x]
    del jan22[x]
    del postcodes[x]

jan12mod = []
comma = ','
for stri in jan12:
    if comma in stri:
        jan12mod.append(int(stri.replace(',', '')))

jan17mod = []
for stri in jan17:
    if comma in stri:
        jan17mod.append(int(stri.replace(',', '')))

jan22mod = []
for stri in jan22:
    if comma in stri:
        jan22mod.append(int(stri.replace(',', '')))


# price growth from 2012 to 2017
priceGrowth = []
# list of percentage growth
for x in range(len(jan12)):
    percentage = float(float(jan17mod[x]) / float(jan12mod[x]))
    priceGrowth.append(percentage)

# price growth from 2017 to 2022
priceGrowth2 = []
for x in range(len(jan12)):
    percentage = float(float(jan22mod[x]) / float(jan17mod[x]))
    priceGrowth2.append(percentage)

# price growth from 2012 to 2022
priceGrowth3 = []
for x in range(len(jan12)):
    percentage = float(float(jan22mod[x]) / float(jan12mod[x]))
    priceGrowth3.append(percentage)


# list ranking all suburbs by price growth from 2012 to 2017
lst = []
for i in range(len(postcodes)):
    lst.append(i + 1)
