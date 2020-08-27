import numpy as np

def save_model(weight , bias):
	model = np.hstack((weight.reshape(-1) , bias))
	np.save('model.npy' , model)
	return

def load_model():
	model = np.load('model.npy')
	weight = model[ : -1].reshape(-1 , 1)
	bias = model[-1]
	return (weight , bias)