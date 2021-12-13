import numpy as np
from math import pow, sin
import matplotlib.pyplot as plt



def init(n_input = 2, n_hidden = 1, n_output = 1):

    w1 = np.random.rand(n_hidden, n_input)
    w2 = np.random.rand(n_output, n_hidden)
    return [w1, w2]

def simulation():
    input_vector = np.random.rand(2,1)
    output = - 12 * input_vector[0] + 5 * input_vector[1] *    
    return [input_vector, output]

def calculus(w, input_vector):
    # this function returns output of layers
    return np.dot(w, input_vector)

def classification(output, NN_out):
    return pow((output - NN_out),2)

def gradient_descent(correct_value, value, input_vector):
    K = 0.03           
    dW =  2 * K * np.dot(np.subtract(correct_value, value),  np.transpose(input_vector))
    return dW

def learning(W,dW):
    # calculates new weigths
    W += dW 
    return W 
              
[w1, w2] = init()
C = []
output_all = []
NN_out_all = []
for i in range(0, 2000):
    [input_vector, output] = simulation()
    L1 = calculus(w1, input_vector)     
    NN_out = calculus(w2, L1) 
    C.append(classification(output, NN_out))
    dw1 = gradient_descent(output, NN_out, input_vector)
    dw2= gradient_descent(output, NN_out, L1)
    learning(w1,dw1)
    learning(w2,dw2) 
    output_all.append(output)
    NN_out_all.append(NN_out[0])
    
plt.plot(C)
plt.show()
plt.plot(output_all)
plt.plot(NN_out_all)
plt.show()


#    print('w2: ', w2)
#    print(pow((output - NN_out),2))


