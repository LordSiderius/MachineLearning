import numpy as np
from math import pow, sin
import matplotlib.pyplot as plt

class Neural_network(object):    

    def __init__(self, nL = 4):
        self.nL = nL
#        num_pnt_L = np.random.randint(1, 5, size = nL)
#        num_pnt_L[0] = 2
#        num_pnt_L[self.nL - 1] = 1
        num_pnt_L = [3, 4, 5, 2]
        self.res_L = [[0,0]] 
        print(self.res_L[-1])    
        self.W = []
        for i in range(0, self.nL - 1):
            devka = np.random.rand(num_pnt_L[i + 1], num_pnt_L[i])
            self.W.append(devka)
            self.res_L.append([0,0])  
                
    def calculus(self, input_vector):
        # this function returns output of layers
        self.res_L[0] = input_vector
        for i in range(0, self.nL - 1):
            self.res_L[i + 1] = np.dot(self.W[i], self.res_L[i])
        return self.res_L[- 1]

    def classification(self, output):
        return np.sum(np.power(np.subtract(output, self.res_L[self.nL - 1]),2))
    
    def gradient_descent(self, correct_value):
        K = 0.005
        for i in range(0, self.nL - 1):
            helpfull_calculation = np.ones(len(self.res_L[i + 1])).reshape(len(self.res_L[i + 1]), 1)
            dW = - 2 * K * np.sum(np.subtract(self.res_L[-1], correct_value)) * np.dot(helpfull_calculation,  np.transpose(self.res_L[i]))
            #learning part 
                       
            self.W[i] = np.add(self.W[i], dW)

def simulation():
    input_vector = np.random.rand(3,1)
    output = np.random.rand(2,1)
    output[0] = - 12 * input_vector[0] + 5 * input_vector[1]
    output[1] = - 12 * input_vector[0] + 5 * input_vector[1] + 1.2 * input_vector[2]
    return [input_vector, output]

c_out = []
all_input_v = []
all_output = []
all_NN_out = []
NN = Neural_network()
for n in range(2000):
    [input_v, output] = simulation()
    all_input_v.append(input_v)
    all_output.append(output[0])
    NN_out = NN.calculus(input_v)
    all_NN_out.append(NN_out[0])
    C = NN.classification(output)
    NN.gradient_descent(output)
    c_out.append(C)

#plt.plot(sim_result)    
plt.plot(all_output)
plt.plot(all_NN_out)
plt.show()
# plt.plot(sim_result[1])
plt.plot(c_out)
plt.show()


