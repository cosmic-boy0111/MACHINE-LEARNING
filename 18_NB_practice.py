from numpy.core import multiarray
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.metrics import confusion_matrix
import seaborn as sn

df = load_wine()

# ['DESCR', 'data', 'feature_names', 'frame', 'target', 'target_names']

inputs = df.data
target = df.target

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
