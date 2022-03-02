'''
1. step
2. sigmoid
3. tanh
4. ReLU
5. Leaky ReLU

'''

import math

def sigmoid(x): 
    return 1/(1+ math.exp(-x))

def tanh(x): 
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def ReLU(x): 
    return max(0,x)

def leaky_relu(x): 
    return max(0.1*x,x)



print(sigmoid(.5))
print(tanh(1))
print(ReLU(1))
print(leaky_relu(-2))