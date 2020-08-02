import argparse
import numpy as np

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_x' , type = str)
	parser.add_argument('--train_y' , type = str)
	args = parser.parse_args()
	return args

def load_data(args):
	with open(args.train_x , 'r') as file:
		next(file)
		train_x = np.array([line.split(',')[1 : ] for line in file] , dtype = float)
	
	with open(args.train_y , 'r') as file:
		next(file)
		train_y = np.array([line.split(',')[1] for line in file] , dtype = int)

	for i in range(2 , 31):
		train_x = np.hstack((train_x , (train_x[ : , 0]**i).reshape((-1 , 1))))

	mean = np.mean(train_x , axis = 0)
	std = np.std(train_x , axis = 0) + 1e-8
	train_x = (train_x - mean) / std

	validation_x = train_x[ : 1000]
	validation_y = train_y[ : 1000]
	train_x = train_x[1000 : ]
	train_y = train_y[1000 : ]

	return (train_x , train_y , validation_x , validation_y , mean , std)

def cov(x):
	number_of_data = x.shape[1]
	mean = np.mean(x , axis = 1)
	return sum((x[ : , i] - mean).reshape((-1 , 1)) @ (x[ : , i] - mean).reshape((1 , -1)) for i in range(number_of_data)) / number_of_data

def generative_model(train_x , train_y):
	class_1 = train_x[np.where(train_y == 1)[0]]
	class_2 = train_x[np.where(train_y == 0)[0]]

	number_of_data = train_x.shape[0]
	number_of_data_1 = class_1.shape[0]
	number_of_data_2 = class_2.shape[0]

	prior_probability_1 = number_of_data_1 / number_of_data
	prior_probability_2 = number_of_data_2 / number_of_data

	mean_1 = np.mean(class_1 , axis = 0).reshape((-1 , 1))
	covariance_1 = cov(class_1.T)

	mean_2 = np.mean(class_2 , axis = 0).reshape((-1 , 1))
	covariance_2 = cov(class_2.T)

	covariance = prior_probability_1 * covariance_1 + prior_probability_2 * covariance_2

	(u , s , vh) = np.linalg.svd(covariance , full_matrices = False)
	inv_covariance = (vh.T * (1 / s)) @ u.T

	weight = inv_covariance @ (mean_1 - mean_2)
	bias = int(-0.5 * (mean_1.T @ inv_covariance @ mean_1) + 0.5 * (mean_2.T @ inv_covariance @ mean_2) + np.log(number_of_data_1 / number_of_data_2))

	return (weight , bias)

def accuracy(x , y , weight , bias):
	count = 0
	number_of_data = x.shape[0]
	for i in range(number_of_data):
		score = x[i] @ weight + bias
		if ((score > 0 and y[i] == 1) or (score < 0 and y[i] == 0)):
			count += 1

	return count / number_of_data

def save_model(mean , std , weight , bias):
	model = np.hstack((weight.reshape((-1)) , bias))
	np.save('generative_mean.npy' , mean)
	np.save('generative_std.npy' , std)
	np.save('generative_model.npy' , model)
	return

def main(args):
	(train_x , train_y , validation_x , validation_y , mean , std) = load_data(args)
	(weight , bias) = generative_model(train_x , train_y)
	print('train accuracy : {:.5f}'.format(accuracy(train_x , train_y , weight , bias)))
	print('validation accuracy : {:.5f}'.format(accuracy(validation_x , validation_y , weight , bias)))
	save_model(mean , std , weight , bias)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)
