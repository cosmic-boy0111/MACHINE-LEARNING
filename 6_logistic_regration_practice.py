import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



df = pd.read_csv('HR_comma_sep.csv')

print(df.head())

dummy1 = pd.get_dummies(df.Department)
dummy2 = pd.get_dummies(df.salary)

final = pd.concat([df,dummy1,dummy2],axis=1)
final.drop(['Department','salary','technical','medium'],axis=1,inplace=True)

x = np.array(final.drop(['left'],axis=1))
y = np.array(final.left)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3)


model = LogisticRegression()

model.fit(np.array(X_train),np.array(y_train))

predict = model.predict(np.array(X_test))

print(model.score(X_test,y_test))

# x = np.array(df.Department)
# y = np.array(df.left)

# plt.bar(x,y)
# plt.show()







