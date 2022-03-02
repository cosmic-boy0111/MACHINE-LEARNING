import pandas as pd 
import math
import numpy as np
from pandas.core.algorithms import mode
from pandas.core.dtypes.missing import notna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

df = pd.read_excel('titanic.xlsx')

age_median = math.floor(df.Age.median())

# print(age_median)

df.Age = df.Age.fillna(age_median)

l_sex = LabelEncoder()

df['Sex'] = l_sex.fit_transform(df['Sex'])

# print(df['Sex'])

X = np.array(df[['Sex','Age','Fare','Pclass']])
y = np.array(df.Survived)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(len(X_train))
print(len(X_test))


# model = DecisionTreeClassifier()
model = RandomForestClassifier(n_estimators=95)
# model = SVC(kernel='rbf')
# model = LogisticRegression()
model.fit(X_train,y_train)

y_predicted = model.predict(X_test)

print(model.score(X_test,y_test))

cm = confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(8,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()











