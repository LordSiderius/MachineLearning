import struct
import gzip
import matplotlib.pyplot as plt
import numpy as np



def read_MNIST(path_data, path_labels, N_samples=200):
	output_list = []

	data = gzip.open(path_data,'r')
	labels = gzip.open(path_labels,'r')

	magic_number_data = struct.unpack('>I', data.read(4))[0]
	N_images = struct.unpack('>I', data.read(4))[0]
	N_rows = struct.unpack('>I', data.read(4))[0]
	N_columns = struct.unpack('>I', data.read(4))[0]

	magic_number_labels = struct.unpack('>I', labels.read(4))[0]
	N_labels = struct.unpack('>I', labels.read(4))[0]


	N_images = N_labels = min(N_samples, N_images)

	data_list = []
	for images in range(N_images):
		sub_list = []

		for row in range(N_rows * N_columns):
			sub_list.append(struct.unpack('>B', data.read(1))[0])


		data_list.append(sub_list)

	labels_list = []

	for label in range(N_labels):
		labels_list.append(struct.unpack('>B', labels.read(1))[0])

	data.close()
	labels.close()

	# print(labels_list[0])
	# image = np.array(data_list[0])  # plot the sample
	# image.shape = (28,28)
	# plt.imshow(image, cmap='gray')
	# plt.show()


	return data_list, labels_list

if __name__ == '__main__':
	path_data = 'data/train-images-idx3-ubyte.gz'
	path_labels = 'data/train-labels-idx1-ubyte.gz'

	read_MNIST(path_data, path_labels)
