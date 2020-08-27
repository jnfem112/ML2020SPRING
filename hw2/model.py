import numpy as np

def cov(x):
	number_of_data = x.shape[1]
	mean = np.mean(x , axis = 1)
	return sum((x[ : , i] - mean).reshape(-1 , 1) @ (x[ : , i] - mean).reshape(1 , -1) for i in range(number_of_data)) / number_of_data

def sigmoid(x):
	return np.clip(1 / (1 + np.exp(-x)) , 1e-6 , 1 - 1e-6)

def save_model(mean , std , weight , bias , method):
	model = np.hstack((weight.reshape(-1) , bias))
	np.save('{}_mean.npy'.format(method) , mean)
	np.save('{}_std.npy'.format(method) , std)
	np.save('{}_model.npy'.format(method) , model)
	return

def load_model(method):
	mean = np.load('{}_mean.npy'.format(method))
	std = np.load('{}_std.npy'.format(method))
	model = np.load('{}_model.npy'.format(method))
	weight = model[ : -1]
	bias = model[-1]
	return (mean , std , weight , bias)