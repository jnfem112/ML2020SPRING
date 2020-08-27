from utils import my_argparse
from data import load_data , dump
from model import load_model

def predict(test_x , weight , bias):
	test_y = list()
	number_of_data = test_x.shape[0]
	for i in range(number_of_data):
		score = test_x[i] @ weight + bias > 0
		test_y.append(1 if (score > 0) else 0)
	return test_y

def main(args):
	(mean , std , weight , bias) = load_model(args.method)
	test_x = load_data(args.test_x , None , args.degree , mean , std)
	test_y = predict(test_x , weight , bias)
	dump(test_y , args.output_file)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)