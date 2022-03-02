# from testing_training import Y_test
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_excel('insurance_data.xlsx')
print(df)

X_train, X_test, y_train, y_test = train_test_split(df[['age']], df['bought'], test_size=0.1)

model = LogisticRegression()

model.fit(X_train,y_train)

print(X_test)
predict = model.predict(X_test)
print(predict)

print(model.score(X_test,y_test))

plt.scatter(X_test,y_test,marker='*',color='red')
plt.plot(X_test,predict)
plt.show()