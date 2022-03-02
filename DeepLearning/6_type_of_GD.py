import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import preprocessing

df  =  pd.read_excel('DeepLearning/homeprices.xlsx')

sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler() 

scaled_X = sx.fit_transform(df.drop('price',axis=1))

print(scaled_X)

scaled_Y = sy.fit_transform(df.price.values.reshape(df.shape[0],1))
# print(df.price.values)
print(scaled_Y)


def batch_gd(X,y_true,epochs,learning_rate = 0.01): 
    number_of_features = X.shape[1]
    w = np.ones(shape=(number_of_features))
    b = 0
    total_sample = X.shape[0]

    cost_list = []
    epochs_list = []

    for i in range(epochs): 

        
        y_predicted = np.dot(w,X.T) + b
        w_grade = -(2/total_sample)*(X.T.dot(y_true-y_predicted))
        b_grade = -(2/total_sample)*np.sum((y_true-y_predicted))

        w = w - learning_rate*w_grade
        b = b - learning_rate*b_grade

        cost = np.mean((np.square(y_true-y_predicted)))

        if i%10==0: 
            cost_list.append(cost)
            epochs_list.append(i)
    
    return w,b,cost,cost_list,epochs_list

w,b,cost,cost_list,epochs_list = batch_gd(scaled_X,scaled_Y.reshape(scaled_Y.shape[0],),500)
print(w,b,cost)



def predict(area,bed,w,b): 
    scaled_x = sx.transform([[area,bed]])
    scaled_price = w[0]*scaled_x[0][0] + w[1]*scaled_x[0][1] + b
    return sy.inverse_transform([[scaled_price]])

print(predict(2600,4,w,b)[0][0])

# plt.xlabel('epoch')
# plt.ylabel('cost')

# plt.plot(epochs_list,cost_list)
# plt.show()


def stochastic_gd(X,y_true,epochs,learning_rate = 0.01): 
    number_of_features = X.shape[1]
    w = np.ones(shape=(number_of_features))
    b = 0
    total_sample = X.shape[0]

    cost_list = []
    epochs_list = []

    for i in range(epochs) : 
        import random 
        idx = random.randint(0,total_sample-1)
        sample_x = X[idx]
        sample_y = y_true[idx]

        y_predict = np.dot(w,sample_x.T) + b

        w_grade = -(2/total_sample)*(sample_x.T.dot(sample_y-y_predict))
        b_grade = -(2/total_sample)*np.sum((sample_y-y_predict))

        w = w - learning_rate*w_grade
        b = b - learning_rate*b_grade

        cost = np.mean((np.square(sample_y-y_predict)))
        if i%100==0: 
            cost_list.append(cost)
            epochs_list.append(i)

    return w,b,cost,cost_list,epochs_list

w_s,b_s,cost_s,cost_list_s,epochs_list_s = stochastic_gd(scaled_X,scaled_Y.reshape(scaled_Y.shape[0],),10000)

print(w_s,b_s,cost_s)

plt.xlabel('epoch')
plt.ylabel('cost')

plt.plot(epochs_list_s,cost_list_s)
plt.show()


