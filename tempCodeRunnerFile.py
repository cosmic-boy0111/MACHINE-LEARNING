import numpy as np 
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import seaborn as sn
from sklearn.metrics import confusion_matrix


df  = load_digits()
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model1 = SVC()
model1.fit(X_train,y_train)
print(model1.score(X_test,y_test))

model2 = SVC(C=10)
model2.fit(X_train,y_train)
print(model2.score(X_test,y_test))

model3 = SVC(kernel='linear')
model3.fit(X_train,y_train)
print(model3.score(X_test,y_test))

model4 = SVC(kernel='rbf')
model4.fit(X_train,y_train)
y_predict = model4.predict(X_test)
print(model4.score(X_test,y_test))

cm = confusion_matrix(y_test,y_predict)
plt.figure(figsize= (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()











