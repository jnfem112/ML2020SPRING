import argparse
import numpy as np
from time import time

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_x' , type = str)
	parser.add_argument('--train_y' , type = str)
	parser.add_argument('--lambd' , type = float)
	parser.add_argument('--learning_rate' , type = float)
	parser.add_argument('--epoch' , type = int)
	args = parser.parse_args()
	return args

def load_data(args):
	with open(args.train_x , 'r') as file:
		next(file)
		train_x = np.array([line.split(',')[1 : ] for line in file] , dtype = float)
	
	with open(args.train_y , 'r') as file:
		next(file)
		train_y = np.array([line.split(',')[1] for line in file] , dtype = int)

	for i in range(2 , 11):
		train_x = np.hstack((train_x , (train_x[ : , 0]**i).reshape((-1 , 1))))

	mean = np.mean(train_x , axis = 0)
	std = np.std(train_x , axis = 0) + 1e-8
	train_x = (train_x - mean) / std

	validation_x = train_x[ : 1000]
	validation_y = train_y[ : 1000]
	train_x = train_x[1000 : ]
	train_y = train_y[1000 : ]

	return (train_x , train_y , validation_x , validation_y , mean , std)

def sigmoid(x):
	return np.clip(1 / (1 + np.exp(-x)) , 1e-6 , 1 - 1e-6)

# Implement Adagrad for logistic regression.
def logistic_regression(train_x , train_y , validation_x , validation_y , args):
	# Initialization.
	(number_of_data , dimension) = train_x.shape
	weight = np.zeros(dimension)
	bias = 0

	# Hyper-parameter.
	lambd = args.lambd
	lr = args.learning_rate
	w_lr = np.ones(dimension)
	b_lr = 0
	epoch = args.epoch

	for i in range(epoch):
		start = time()

		z = np.dot(train_x , weight) + bias
		pred = sigmoid(z)
		loss = train_y - pred

		# Calculate gradient.
		w_grad = -1 * np.dot(loss , train_x) + 2 * lambd * weight
		b_grad = -1 * np.sum(loss)

		# Update weight and bias.
		w_lr += w_grad**2
		b_lr += b_grad**2
		bias = bias - lr / np.sqrt(b_lr) * b_grad
		weight = weight - lr / np.sqrt(w_lr) * w_grad

		# Calculate loss and accuracy.
		loss = -1 * np.mean(train_y * np.log(pred + 1e-100) + (1 - train_y) * np.log(1 - pred + 1e-100))
		train_accuracy = accuracy(train_x , train_y , weight , bias)
		validation_accuracy = accuracy(validation_x , validation_y , weight , bias)

		end = time()
		print('({}s) [epoch {}/{}] loss : {:.5f} , train accuracy : {:.5f} , validation accuracy : {:.5f}'.format(int(end - start) , i + 1 , epoch , loss , train_accuracy , validation_accuracy))

	return (weight , bias)

def accuracy(x , y , weight , bias):
	count = 0
	number_of_data = x.shape[0]
	for i in range(number_of_data):
		probability = sigmoid(weight @ x[i] + bias)
		if ((probability > 0.5 and y[i] == 1) or (probability < 0.5 and y[i] == 0)):
			count += 1
	return count / number_of_data

def save_model(mean , std , weight , bias):
	model = np.hstack((weight , bias))
	np.save('logistic_mean.npy' , mean)
	np.save('logistic_std.npy' , std)
	np.save('logistic_model.npy' , model)
	return

def main(args):
	(train_x , train_y , validation_x , validation_y , mean , std) = load_data(args)
	(weight , bias) = logistic_regression(train_x , train_y , validation_x , validation_y , args)
	save_model(mean , std , weight , bias)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)
