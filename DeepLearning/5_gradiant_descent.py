import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


df = pd.read_excel('DeepLearning/insurance_data2.xlsx')
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df[['age','affordibility']],df.bought_insurance, test_size=0.2, random_state=25)

print(X_train.shape)
print(len(X_train))

X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age']/100;


X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age']/100;

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(2,)),
        keras.layers.Dense(1,activation='sigmoid',kernel_initializer='ones',bias_initializer = 'zero')
    ]
)

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_scaled,y_train,epochs=5000)

print(model.evaluate(X_test_scaled,y_test))

y_predict = model.predict(X_test_scaled)
print(y_test)
print(y_predict)

def check(x):
    if x >=0.5: 
        return 1
    return 0

y_predict_new = [ check(i) for i in y_predict]

cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predict_new)

import seaborn as sn 
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


plt.show()

coef, intercept = model.get_weights()
print(coef,intercept)






