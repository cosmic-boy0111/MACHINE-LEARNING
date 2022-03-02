import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import data
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']

diabetes_X = diabetes.data[:,np.newaxis,2]

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)

diabetes_Y_predicted =   model.predict(diabetes_X_test)

print("mean squared error is : ", mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))

print('weights : ',model.coef_)
print('intercept : ',model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()

# mean squared error is :  3035.0601152912695
# weights :  [941.43097333]
# intercept :  153.39713623331698
