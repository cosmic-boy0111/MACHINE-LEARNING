
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model


data = pd.read_excel('l_regration_exc.xlsx')

model = linear_model.LinearRegression()

model.fit(np.array(data[['year']]),np.array(data.income))

pred = model.predict([[2020]])


plt.plot(np.array(data[['year']]),np.array(data.income),color='red')

plt.scatter([[2020]],pred,color='blue')

plt.show()
print(pred)



