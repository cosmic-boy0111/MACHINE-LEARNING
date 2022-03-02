from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sn



df = load_iris()

X = df.data 
y = df.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()

model.fit(X_train,y_train)

y_predicted = model.predict(X_test)

print(model.score(X_test,y_test))

cm = confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(8,5))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()

