import argparse
import pandas as pd
import numpy as np

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_data' , type = str)
	parser.add_argument('--output_file' , type = str)
	args = parser.parse_args()
	return args

def load_data(args):
	df = pd.read_csv(args.test_data , encoding = 'big5' , header = None)

	df[df == 'NR'] = 0

	data = df.values
	data = np.delete(data , [0 , 1] , axis = 1)
	data = data.astype(np.float)

	(number_of_row , number_of_column) = data.shape
	data = np.hstack([data[18 * i : 18 * (i + 1)] for i in range(number_of_row // 18)])

	test_x = list()
	(number_of_row , number_of_column) = data.shape
	for i in range(number_of_column // 9):
		x = data[ : , 9 * i : 9 * (i + 1)].reshape(-1)
		test_x.append(x)
	test_x = np.array(test_x)

	return test_x

def load_model():
	model = np.load('model.npy')
	weight = model[ : -1].reshape((-1 , 1))
	bias = model[-1]
	return (weight , bias)

def predict(test_x , weight , bias):
	return test_x @ weight + bias

def dump(test_y , args):
	number_of_data = test_y.shape[0]
	df = pd.DataFrame({'id' : ['id_{}'.format(i) for i in range(number_of_data)] , 'value' : test_y.reshape(-1)})
	df.to_csv(args.output_file , index = False)
	return

def main(args):
	test_x = load_data(args)
	(weight , bias) = load_model()
	test_y = predict(test_x , weight , bias)
	dump(test_y , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)