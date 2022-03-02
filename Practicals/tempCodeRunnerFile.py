import numpy as np 
from sklearn.datasets import fetch_olivetti_faces
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt 
import pickle
import pandas as pd

olivetti = fetch_olivetti_faces()

print(dir(olivetti))
X = np.array(olivetti.data)
y = np.array(olivetti.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)


with open('olivetti_model','wb') as f: 
    pickle.dump(model,f)

with open('olivetti_model','rb') as f: 
    om = pickle.load(f)

y_predict = om.predict(X_test)

print(y_test)
print(y_predict)

print(model.score(X_test,y_test)*100)

cm = confusion_matrix(y_test,y_predict)
plt.figure(figsize= (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()















