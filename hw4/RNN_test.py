import argparse
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import torch
from torch.utils.data import Dataset , DataLoader
from RNN import RNN
import torch.nn as nn

class Vocabulary():
	def __init__(self):
		W2Vmodel = Word2Vec.load('Word2Vec.model')
		self.word2vector = {token : W2Vmodel.wv[token] for token in W2Vmodel.wv.vocab}
		self.word2vector['<PAD>'] = np.zeros(W2Vmodel.vector_size)
		self.word2vector['<UNK>'] = np.zeros(W2Vmodel.vector_size)
		self.token2index = {token : index for (index , token) in enumerate(self.word2vector)}
		self.index2token = {index : token for (index , token) in enumerate(self.word2vector)}
		self.embedding = torch.FloatTensor([self.word2vector[token] for token in self.word2vector])
		return

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--test_data' , type = str)
	parser.add_argument('--output_file' , type = str)
	parser.add_argument('--max_length' , type = int)
	parser.add_argument('--batch_size' , type = int)
	args = parser.parse_args()
	return args

def preprocess(data , vocabulary , args):
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
	data = trim_and_pad(data , vocabulary , args.max_length)

	return np.array(data)

def load_data(args):
	vocabulary = Vocabulary()

	def valid(token):
		return all(character.isalnum() for character in token)

	test_x = list()
	with open(args.test_data , 'r') as file:
		next(file)
		for line in file:
			line = ','.join(line.split(',')[1 : ])
			line = line.rstrip('\n').split()
			x = [token for token in line if valid(token)]
			test_x.append(x)

	test_x = preprocess(test_x , vocabulary , args)

	return (test_x , vocabulary)

def test(test_x , model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size

	test_loader = DataLoader(test_x , batch_size = batch_size , shuffle = False , num_workers = 8)

	model.to(device)
	model.eval()
	test_y = list()
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device , dtype = torch.long)
			y = model(data)
			y = y.squeeze()
			index = (y >= 0.5).int()
			test_y.append(index.cpu().detach().numpy())
	test_y = np.concatenate(test_y , axis = 0)

	return test_y

def dump(test_y , args):
	number_of_data = test_y.shape[0]
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : test_y})
	df.to_csv(args.output_file , index = False)
	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(test_x , vocabulary) = load_data(args)
	model = RNN(vocabulary.embedding , vocabulary.token2index['<PAD>'])
	model.load_state_dict(torch.load('RNN.pkl' , map_location = device))
	test_y = test(test_x , model , device , args)
	dump(test_y , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)
