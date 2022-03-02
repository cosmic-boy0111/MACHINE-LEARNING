'''
svm = support vector machine

determine on the basis of partition of data


take a line that separate the data and produce
the highest margin which help to easy classify the different data


parameters : 
1. Gamma (calculate margin)
    --> high gamma
    --> low gamma
2. Regularization(C)
    --> low
    --> high


kernel: 


'''

import pandas as pd
from pandas.core.algorithms import mode
from pandas.core.frame import DataFrame 
from sklearn.datasets import load_iris 
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ['DESCR', 'data', 'feature_names', 'filename', 'frame', 'target', 'target_names']

iris = load_iris()

print(iris.feature_names)

df = pd.DataFrame(iris.data)

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


####### regularization ###########

model = SVC()

model.fit(X_train,y_train)

# print(X_test)
print(model.score(X_test,y_test))

y_predict = model.predict(X_test)
print(y_test)
print(y_predict)

model_C = SVC(C=10)

model_C.fit(X_train,y_train)


print(model_C.score(X_test,y_test))




######### gamma ##############

model_g = SVC(gamma=1)
model_g.fit(X_train,y_train)

print(model_g.score(X_test,y_test))



########### kernel ############

model_linear_kernel = SVC(kernel='linear')
model_linear_kernel.fit(X_train,y_train)

print(model_linear_kernel.score(X_test,y_test))




# plt.xlabel('sepal length')
# plt.ylabel('sepal width')

# # plt.scatter(df0[0],df0[1],color='green',marker="+")
# # plt.scatter(df1[0],df1[1],color='blue',marker=".")
# plt.scatter(df0[2],df0[3],color='green',marker="+")
# plt.scatter(df1[2],df1[3],color='blue',marker=".")

# plt.show()








