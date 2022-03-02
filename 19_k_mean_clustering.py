from operator import irshift
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)
print(y_test)
print(y_predict)


print(model.score(X_test,y_test)*100)

cm = confusion_matrix(y_test,y_predict)
plt.figure(figsize= (7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()



