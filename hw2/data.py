import pandas as pd
import numpy as np

def load_data(path_x , path_y , degree , mean = None , std = None , split = 1000):
	if ('train' in path_x):
		with open(path_x , 'r') as file:
			next(file)
			train_x = np.array([line.split(',')[1 : ] for line in file] , dtype = float)
		
		with open(path_y , 'r') as file:
			next(file)
			train_y = np.array([line.split(',')[1] for line in file] , dtype = int)

		for i in range(2 , degree + 1):
			train_x = np.hstack((train_x , (train_x[ : , 0]**i).reshape(-1 , 1)))

		mean = np.mean(train_x , axis = 0)
		std = np.std(train_x , axis = 0) + 1e-8
		train_x = (train_x - mean) / std

		return (train_x[split : ] , train_y[split : ] , train_x[ : split] , train_y[ : split] , mean , std)
	else:
		with open(path_x , 'r') as file:
			next(file)
			test_x = np.array([line.split(',')[1 : ] for line in file] , dtype = float)

		for i in range(2 , degree + 1):
			test_x = np.hstack((test_x , (test_x[ : , 0]**i).reshape(-1 , 1)))

		test_x = (test_x - mean) / std

		return test_x

def dump(predict , output_file):
	number_of_data = len(predict)
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : predict})
	df.to_csv(output_file , index = False)
	return