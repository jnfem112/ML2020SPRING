import argparse
import numpy as np
import pandas as pd

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_x' , type = str)
	parser.add_argument('--output_file' , type = str)
	args = parser.parse_args()
	return args

def load_model():
	mean = np.load('generative_mean.npy')
	std = np.load('generative_std.npy')
	model = np.load('generative_model.npy')
	weight = model[ : -1].reshape((-1 , 1))
	bias = model[-1]
	return (mean , std , weight , bias)

def load_data(args , mean , std):
	with open(args.test_x , 'r') as file:
		next(file)
		test_x = np.array([line.split(',')[1 : ] for line in file] , dtype = float)

	for i in range(2 , 31):
		test_x = np.hstack((test_x , (test_x[ : , 0]**i).reshape((-1 , 1))))

	test_x = (test_x - mean) / std

	return test_x

def predict(test_x , weight , bias):
	test_y = list()
	number_of_data = test_x.shape[0]
	for i in range(number_of_data):
		test_y.append(1 if (test_x[i] @ weight + bias > 0) else 0)
	return test_y

def dump(test_y , args):
	number_of_data = len(test_y)
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : test_y})
	df.to_csv(args.output_file , index = False)
	return

def main(args):
	(mean , std , weight , bias) = load_model()
	test_x = load_data(args , mean , std)
	test_y = predict(test_x , weight , bias)
	dump(test_y , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)