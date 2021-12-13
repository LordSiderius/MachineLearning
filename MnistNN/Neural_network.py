import numpy as np
import math
from MNIST_reader import read_MNIST
import matplotlib.pyplot as plt

class MLP(object):

    def __init__(self, input_size, output_size, n_hidden=1, num_pnt_L=[16]):
        self.Layer = []
        self.LayerError = []
        self.LayerRaw = []
        self.lay_count = n_hidden + 2
        self.Layer.append(np.zeros((input_size, 1)))
        self.LayerError.append(np.zeros((input_size, 1)))
        self.LayerRaw.append(np.zeros((input_size, 1)))
        for layer in range(n_hidden):
            self.Layer.append(np.zeros((num_pnt_L[layer - 1], 1)))
            self.LayerError.append(np.zeros((num_pnt_L[layer - 1], 1)))
            self.LayerRaw.append(np.zeros((num_pnt_L[layer - 1], 1)))
        self.Layer.append(np.zeros((output_size, 1)))
        self.LayerError.append(np.zeros((output_size, 1)))
        self.LayerRaw.append(np.zeros((output_size, 1)))

    #     weights definition
        self.W = []
        self.W.append(np.ones((num_pnt_L[0], input_size))*0.5)
        for weights in range(n_hidden - 1):
            self.W.append(np.ones((num_pnt_L[weights + 1], num_pnt_L[weights]))*0.5)
        self.W.append(np.ones((output_size, num_pnt_L[-1]))*0.5)

    #     bias definition
        self.b = []
        for layer in range(n_hidden):
            self.b.append(np.ones((num_pnt_L[layer - 1], 1)))
        self.b.append(np.ones((output_size, 1)))


    def calculus(self, input_vector):

        self.LayerRaw[0] = np.divide(np.array(input_vector), 255)
        self.LayerRaw[0] = self.Layer[0].reshape((len(input_vector), 1))

        for i in range(self.lay_count - 1):
            self.LayerRaw[i + 1] = np.matmul(self.W[i], self.Layer[i])
            self.Layer[i + 1] = self.LayerRaw[i + 1]

        return self.Layer[-1]

    def classification(self, output):
        output = output.reshape((output.shape[0], 1))
        class_output = np.sum(np.power(np.subtract(output, self.Layer[-1]), 2)) / output.shape[0]
        return class_output

    def learn(self, correct_value, alpha=0.00001, kick=False):
        # point errors must be calcualted for each lazer
        correct_value = np.array(correct_value)
        correct_value = correct_value.reshape((correct_value.shape[0], 1))
        self.Layer[-1] = self.Layer[-1].reshape((correct_value.shape[0], 1))
        # self.LayerError[-1] = alpha * np.multiply(np.subtract(correct_value, self.Layer[-1]), self.act_fnc_der(self.Layer[-1]))
        self.LayerError[-1] = np.multiply(np.subtract(correct_value, self.Layer[-1]), alpha)
        dW = np.transpose(np.matmul(self.Layer[-2], np.transpose(self.LayerError[-1])))
        self.W[-1] = np.add(self.W[-1], dW)


        for i in range(1, self.lay_count - 1):
            self.LayerError[-i - 1] = np.matmul(np.transpose(self.W[-i]), self.LayerError[-i])
            # self.LayerError[-i - 1] = np.multiply(self.LayerError[-i - 1], self.act_fnc_der(self.Layer[-i - 1]))
            dW = np.transpose(np.matmul(self.Layer[-i - 2], np.transpose(self.LayerError[-i - 1])))
            self.W[-i - 1] = np.add(self.W[-i - 1], dW)

        if kick:
            for j in range(self.lay_count - 1):
                self.W[j] = np.add(self.W[j], 0.005)

        pass

    def act_fnc(self, value):

        value = np.clip(value, -1000, 1000)
        output = 1/(1 + np.power(math.e, -value))

        return output

    def act_fnc_der(self, value):

        output = self.act_fnc(value) * (1 - self.act_fnc(value))

        return output



if __name__ == '__main__':
    samples = 1
    path_data = 'data/train-images-idx3-ubyte.gz'
    path_labels = 'data/train-labels-idx1-ubyte.gz'

    data_list, labels_list = read_MNIST(path_data, path_labels, samples)

    mlp = MLP(len(data_list[0]), 10)
    print(data_list[0])
    print(mlp.calculus(data_list[0]))



    # print(labels_list[1])
    # image = np.array(data_list[1])  # plot the sample
    # image.shape = (28,28)
    # plt.imshow(image, cmap='gray')
    # plt.show()


    # mlp = MLP(len(data_list[0]), 10)
    # alpha = 0.0005
    # error = []
    # kick = False
    # print(mlp.act_fnc([-1000000000,0]))
    # for epochs in range(1):
    #     results = []
    #     for i in range(samples):
    #         if i % 100 == 0:
    #             kick = True
    #         else:
    #             kick = False
    #         correct_value = np.zeros((10, 1))
    #         correct_value[labels_list[i]] = 1
    #         results.append(mlp.calculus(data_list[i]))
    #         error.append(mlp.classification(correct_value))
    #         mlp.learn(correct_value, alpha, kick)
    #
    # output_numbers = [np.argmax(results[i]) for i in range(len(results))]
    #
    # plt.plot(output_numbers)
    # plt.plot(labels_list[0:samples])
    #
    # for k in range(samples):
    #     pos_count = 0
    #     if labels_list[k] == output_numbers[k]:
    #         pos_count += 1
    # percentage = pos_count / samples
    #
    # print(percentage)
    # # print(labels_list[15])
    # # image = np.array(data_list[0])  # plot the sample
    # # image.shape = (28,28)
    # # plt.imshow(image, cmap='gray')
    # # plt.show()
    #
    # plt.show()
    # plt.plot(error)
    # plt.show()
    # # #
    # # # mlp.learn([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
