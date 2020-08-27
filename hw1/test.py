from utils import my_argparse
from data import load_data , dump
from model import load_model

def predict(test_x , weight , bias):
	return test_x @ weight + bias

def main(args):
	print(args.test_data)
	test_x = load_data(args.test_data)
	(weight , bias) = load_model()
	test_y = predict(test_x , weight , bias)
	dump(test_y , args.output_file)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)