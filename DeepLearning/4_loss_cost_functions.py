'''
1. sparse_categorical_crossentropy
2. binary_crossentropy
3. categorical_crossentropy
4. mean_squared_error
5. mean_absolute_error

'''


import numpy as np 

y_predicted = np.array([1,1,0,0,1])
y_true = ([0.30,0.7,1,0,0.5])

#### mean absolute error

def mae(y_true,y_predicted): 
    total_err = 0
    for yt, yp in zip(y_true,y_predicted): 
        total_err += abs(yt-yp)
    print("Total Error",total_err)

    return total_err/len(y_true)

print( mae(y_true,y_predicted) )

print(np.mean(np.abs(y_true-y_predicted)))
print(np.mean((y_true-y_predicted)*(y_true-y_predicted)))


##### log loss / binary_crossentropy


def log_loss(y_true,y_predicted): 
    total_err = 0
    for yt, yp in zip(y_true,y_predicted): 
        total_err += (yt*np.log(yp)+(1-yt)*np.log(1-yp))

    return -total_err/len(y_true)

epsilon = 1e-15

y_predicted_new = [ max(i,epsilon) for i in y_predicted ]
y_predicted_new = [ min(i,1-epsilon) for i in  y_predicted_new]

y_predicted_new = np.array(y_predicted_new)

print(log_loss(y_true,y_predicted_new))













