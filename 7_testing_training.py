import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_excel('carprices.xlsx')
# print(df.head())

X = df[['mileage','age']]
Y = df['price']

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3)

model = LinearRegression()
model.fit(X_train,Y_train)

predict = model.predict(X_test)











