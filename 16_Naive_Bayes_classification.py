import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


df = pd.read_excel('./titanic.xlsx')

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1,inplace=True)

target = df.Survived
inputs = df.drop(['Survived'],axis=1)

dummies = pd.get_dummies(df.Sex)

inputs = pd.concat([inputs,dummies],axis=1)
inputs.drop(['Sex'],axis=1,inplace=True)

inputs = inputs.fillna(inputs.Age.mean())

# print(inputs)
# print(target)

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

model = GaussianNB()
model.fit(X_train,y_train)

print(model.score(X_test,y_test))

y_predict = model.predict(X_test)

cm = confusion_matrix(y_test,y_predict)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


