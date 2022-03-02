import pandas as pd 
import numpy as np
import math
from pandas.core.algorithms import mode
from sklearn import linear_model



data = pd.read_excel('multiple_attributes_homeprices.xlsx')

print(data)
median_bedrooms =  math.floor( data.bedrooms.median() )

print(median_bedrooms)

data.bedrooms =  data.bedrooms.fillna(median_bedrooms)
print(data)

model = linear_model.LinearRegression()

model.fit(data[['area','bedrooms','age']],data.price)

print(model.coef_)
print(model.intercept_)

predict = model.predict([[2500,4,5]])

print(predict)

