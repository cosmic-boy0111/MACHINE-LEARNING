import pandas as pd
from pandas.io.formats.format import CategoricalFormatter 



df = pd.read_excel('for_dummy_variable.xlsx')
# print(df)
dummies = pd.get_dummies(df.town)
# print(dummies)

df_dummies =  pd.concat([df,dummies],axis=1)
df_dummies.drop(['town','west windsor'],axis=1,inplace=True)

X = df_dummies.drop(['price'],axis=1)
print(X)

Y = df_dummies.price
print(Y)

print(df_dummies)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,Y)

predict =  model.predict([[3400,0,0]])
print(predict)
predict =  model.predict([[2800,0,1]])
print(predict)
predict =  model.predict([[2800,1,0]])
print(predict)






