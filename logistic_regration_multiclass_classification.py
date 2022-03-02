import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits = load_digits()

# print(dir(digits))
# print(digits.images[0])

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
model = LogisticRegression()

model.fit(X_train,y_train)

predict = model.predict(X_test)

# print(y_test)
# print(predict)
# print(model.score(X_test,y_test))


print(predict)
cm = confusion_matrix(y_test,predict)
plt.figure(figsize= (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()


# plt.matshow(digits.images[0])
# plt.show()






