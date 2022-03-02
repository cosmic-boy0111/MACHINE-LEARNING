import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
from sklearn import linear_model
import pickle


df = pd.read_excel('homeprices_train.xlsx')


model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)

with open('model_pickle','wb') as f:
    pickle.dump(model,f)

with open('model_pickle','rb') as f: 
    mp = pickle.load(f)

d = pd.read_excel('homeprices_test.xlsx')

pred = mp.predict([[3000]])

print(pred)

print('weights : ',mp.coef_)
print('intercept : ',mp.intercept_)
