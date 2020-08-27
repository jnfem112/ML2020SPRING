import pandas as pd
import numpy as np
from utils import Vocabulary

def valid(token):
	return all(character.isalnum() for character in token)

def load_vocabulary(train_data , nolabel_data , test_data):
	vocabulary = list()

	with open(train_data , 'r') as file:
		for line in file:
			line = line.rstrip('\n').split()[2 : ]
			line = [token for token in line if valid(token)]
			vocabulary.append(line)

	with open(nolabel_data , 'r') as file:
		for line in file:
			line = line.rstrip('\n').split()
			line = [token for token in line if valid(token)]
			vocabulary.append(line)

	with open(test_data , 'r') as file:
		next(file)
		for line in file:
			line = ','.join(line.split(',')[1 : ])
			line = line.rstrip('\n').split()
			line = [token for token in line if valid(token)]
			vocabulary.append(line)

	return vocabulary

def load_data(train_data , nolabel_data , test_data , max_length , split = 20000):
	vocabulary = Vocabulary()

	if (train_data is not None):
		train_x = list()
		train_y = list()
		with open(train_data , 'r') as file:
			for line in file:
				line = line.rstrip('\n').split()
				x = [token for token in line[2 : ] if valid(token)]
				y = int(line[0])
				train_x.append(x)
				train_y.append(y)

		nolabel_x = list()
		with open(nolabel_data , 'r') as file:
			for line in file:
				line = line.rstrip('\n').split()
				x = [token for token in line if valid(token)]
				nolabel_x.append(x)

		train_x = preprocess(train_x , vocabulary , max_length)
		nolabel_x = preprocess(nolabel_x , vocabulary , max_length)

		return (train_x[split : ] , train_y[split : ] , nolabel_x , train_x[ : split] , train_y[ : split] , vocabulary)
	else:
		test_x = list()
		with open(test_data , 'r') as file:
			next(file)
			for line in file:
				line = ','.join(line.split(',')[1 : ])
				line = line.rstrip('\n').split()
				x = [token for token in line if valid(token)]
				test_x.append(x)

		test_x = preprocess(test_x , vocabulary , max_length)

		return (test_x , vocabulary)

def preprocess(data , vocabulary , max_length):
	def token2index(data , vocabulary):
		for i in range(len(data)):
			for j in range(len(data[i])):
				data[i][j] = vocabulary.token2index[data[i][j]] if (data[i][j] in vocabulary.word2vector) else vocabulary.token2index['<UNK>']
		return data

	def trim_and_pad(data , vocabulary , max_length):
		for i in range(len(data)):
			data[i] = data[i][ : min(len(data[i]) , max_length)]
			data[i] += (max_length - len(data[i])) * [vocabulary.token2index['<PAD>']]
		return data

	data = token2index(data , vocabulary)
	data = trim_and_pad(data , vocabulary , max_length)
	return np.array(data)

def dump(predict , output_file):
	number_of_data = predict.shape[0]
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : predict})
	df.to_csv(output_file , index = False)
	return