import numpy as np
from time import time
from utils import my_argparse
from data import load_data
from model import cov , sigmoid , save_model

def generative_model(train_x , train_y , validation_x , validation_y):
	class_1 = train_x[np.where(train_y == 1)[0]]
	class_2 = train_x[np.where(train_y == 0)[0]]
	number_of_data = train_x.shape[0]
	number_of_data_1 = class_1.shape[0]
	number_of_data_2 = class_2.shape[0]

	prior_probability_1 = number_of_data_1 / number_of_data
	prior_probability_2 = number_of_data_2 / number_of_data
	mean_1 = np.mean(class_1 , axis = 0).reshape(-1 , 1)
	covariance_1 = cov(class_1.T)
	mean_2 = np.mean(class_2 , axis = 0).reshape(-1 , 1)
	covariance_2 = cov(class_2.T)
	covariance = prior_probability_1 * covariance_1 + prior_probability_2 * covariance_2

	(u , s , vh) = np.linalg.svd(covariance , full_matrices = False)
	inv_covariance = (vh.T * (1 / s)) @ u.T
	weight = inv_covariance @ (mean_1 - mean_2)
	bias = int(-0.5 * (mean_1.T @ inv_covariance @ mean_1) + 0.5 * (mean_2.T @ inv_covariance @ mean_2) + np.log(number_of_data_1 / number_of_data_2))

	print('train accuracy : {:.5f}'.format(accuracy(train_x , train_y , weight , bias)))
	print('validation accuracy : {:.5f}'.format(accuracy(validation_x , validation_y , weight , bias)))
	return (weight , bias)

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
		score = x[i] @ weight + bias
		if ((score > 0 and y[i] == 1) or (score < 0 and y[i] == 0)):
			count += 1
	return count / number_of_data

def main(args):
	(train_x , train_y , validation_x , validation_y , mean , std) = load_data(args.train_x , args.train_y , args.degree)
	if (args.method == 'generative'):
		(weight , bias) = generative_model(train_x , train_y , validation_x , validation_y)
	else:
		(weight , bias) = logistic_regression(train_x , train_y , validation_x , validation_y , args)
	save_model(mean , std , weight , bias , args.method)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)