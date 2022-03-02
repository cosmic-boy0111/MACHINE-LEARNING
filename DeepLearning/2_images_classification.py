import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

(X_train,y_train) , (X_test,y_test) = keras.datasets.cifar10.load_data()

X_train = (X_train)/255 
X_test = (X_test )/255



model = keras.Sequential(
    [
        # keras.layers.Dense(10,input_shape=(32,784),activation='sigmoid')
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(100,activation='relu'),
        keras.layers.Dense(10,activation='sigmoid')
    ]
)

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


model.fit(X_train,y_train,epochs=10)


model.evaluate(X_test,y_test)

y_predict = model.predict(X_test)

print(y_test)
print(y_predict)
print( np.argmax( y_predict[1]))

plt.figure(figsize=(7,1))
plt.imshow(X_test[1])
plt.show()

y_predict_labels = [ np.argmax(i) for i in y_predict]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predict_labels)

import seaborn as sn 
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()










