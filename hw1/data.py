import pandas as pd
import numpy as np

def load_data(path , split = 500):
	df = pd.read_csv(path , encoding = 'big5' , header = 'infer' if ('train' in path) else None)

	df[df == 'NR'] = 0

	data = df.values
	data = np.delete(data , [0 , 1 , 2] if ('train' in path) else [0 , 1] , axis = 1)
	data = data.astype(np.float)

	(number_of_row , number_of_column) = data.shape
	data = np.hstack([data[18 * i : 18 * (i + 1)] for i in range(number_of_row // 18)])

	if ('train' in path):
		train_x = list()
		train_y = list()
		(number_of_row , number_of_column) = data.shape
		for i in range(number_of_column - 9):
			x = data[ : , i : i + 9].reshape(-1)
			y = data[9 , i + 9]
			train_x.append(x)
			train_y.append(y)
		train_x = np.array(train_x)
		train_y = np.array(train_y)
		return (train_x[split : ] , train_y[split : ] , train_x[ : split] , train_y[ : split])
	else:
		test_x = list()
		(number_of_row , number_of_column) = data.shape
		for i in range(number_of_column // 9):
			x = data[ : , 9 * i : 9 * (i + 1)].reshape(-1)
			test_x.append(x)
		test_x = np.array(test_x)
		return test_x

def dump(predict , path):
	number_of_data = predict.shape[0]
	df = pd.DataFrame({'id' : ['id_{}'.format(i) for i in range(number_of_data)] , 'value' : predict.reshape(-1)})
	df.to_csv(path , index = False)
	return