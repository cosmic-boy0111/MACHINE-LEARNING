import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
from sklearn import linear_model
import seaborn as sn
from sklearn.metrics import confusion_matrix


df = pd.read_excel('homeprices_train.xlsx')


# d = {
#     'area' : [],
#     'price' : []
# }

# n = int(input('enter data size : '))
# print('enter area and price : ')


# for _ in range(n): 
#     x,y = map(int,input().split(' '))
#     d['area'].append(x)
#     d['price'].append(y)

# df = pd.DataFrame.from_dict(d)

#  area   price
# 0  2600  550000
# 1  3000  565000
# 2  3200  610000
# 3  3600  680000
# 4  4000  725000
print(df)


model = linear_model.LinearRegression(
    fit_intercept=True, 
    normalize=False, 
    copy_X=True, 
    n_jobs=None, 
    positive=False
)
model.fit(df[['area']],df.price)

d = pd.read_excel('homeprices_test.xlsx')

pred = model.predict(df[['area']])

print('weights : ',model.coef_)
print('intercept : ',model.intercept_)

plt.xlabel('area (sqr ft)')
plt.ylabel('price (Rs)')
plt.grid()


plt.scatter(df.area,df.price,color='red',marker='*')
plt.plot(df.area,pred,color='blue')

plt.show()


