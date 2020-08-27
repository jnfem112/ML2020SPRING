import argparse

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_x' , type = str , default = 'X_train')
	parser.add_argument('--train_y' , type = str , default = 'Y_train')
	parser.add_argument('--test_x' , type = str , default = 'X_test')
	parser.add_argument('--output_file' , type = str , default = 'predict.csv')
	parser.add_argument('--method' , type = str , default = 'logistic')
	parser.add_argument('--degree' , type = int , default = 10)
	parser.add_argument('--learning_rate' , type = float , default = 0.05)
	parser.add_argument('--lambd' , type = float , default = 0.001)
	parser.add_argument('--epoch' , type = int , default = 1000)
	args = parser.parse_args()
	return args