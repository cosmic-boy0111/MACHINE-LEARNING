import matplotlib.pyplot as plt
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle


iris = load_iris()

print(dir(iris))

print(iris.target_names)

X = np.array( iris.data )
y = np.array( iris.target )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()

model.fit(X_train,y_train)

with open('iris_model','wb') as f: 
    pickle.dump(model,f)

with open('iris_model','rb') as f: 
    im = pickle.load(f)

y_predict = im.predict(X_test)

print(y_test)
print(y_predict)

print(model.score(X_test,y_test)*100)

cm = confusion_matrix(y_test,y_predict)
plt.figure(figsize= (7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()



