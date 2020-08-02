import argparse
import os
import numpy as np
from gensim.models import Word2Vec
import torch
from torch.utils.data import Dataset , DataLoader
from RNN import RNN
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from time import time

max_accuracy = 0

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

class dataset(Dataset):
	def __init__(self , data , label):
		self.data = data
		self.label = label
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		if (self.label is not None):
			return (torch.tensor(self.data[index]) , self.label[index])
		else:
			return torch.tensor(self.data[index])

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--train_data' , type = str)
	parser.add_argument('--nolabel_data' , type = str)
	parser.add_argument('--max_length' , type = int)
	parser.add_argument('--batch_size' , type = int)
	parser.add_argument('--learning_rate' , type = float)
	parser.add_argument('--epoch' , type = int)
	parser.add_argument('--threshold_1' , type = float)
	parser.add_argument('--threshold_2' , type = float)
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

	train_x = list()
	train_y = list()
	with open(args.train_data , 'r') as file:
		for line in file:
			line = line.rstrip('\n').split()
			x = [token for token in line[2 : ] if valid(token)]
			y = int(line[0])
			train_x.append(x)
			train_y.append(y)

	nolabel_x = list()
	with open(args.nolabel_x , 'r') as file:
		for line in file:
			line = line.rstrip('\n').split()
			x = [token for token in line if valid(token)]
			nolabel_x.append(x)

	train_x = preprocess(train_x , vocabulary , args)
	nolabel_x = preprocess(nolabel_x , vocabulary , args)

	validation_x = train_x[ : 20000]
	validation_y = train_y[ : 20000]
	train_x = train_x[20000 : ]
	train_y = train_y[20000 : ]

	return (train_x , train_y , nolabel_x , validation_x , validation_y , vocabulary)

def train(train_x , train_y , validation_x , validation_y , model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	epoch = args.epoch

	global max_accuracy

	train_dataset = dataset(train_x , train_y)
	validation_dataset = dataset(validation_x , validation_y)
	train_loader = DataLoader(train_dataset , batch_size = batch_size , shuffle = True , num_workers = 8)
	validation_loader = DataLoader(validation_dataset , batch_size = batch_size , shuffle = False , num_workers = 8)

	model.to(device)
	optimizer = Adam(model.parameters() , lr = learning_rate)
	criterion = nn.BCELoss()
	for i in range(epoch):
		start = time()
		model.train()
		count = 0
		total_loss = 0
		for (j , (data , label)) in enumerate(train_loader):
			(data , label) = (data.to(device , dtype = torch.long) , label.to(device , dtype = torch.float))
			optimizer.zero_grad()
			y = model(data)
			y = y.squeeze()
			index = (y >= 0.5).int()
			count += torch.sum(label == index).item()
			loss = criterion(y , label)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()

			if (j < len(train_loader) - 1):
				n = (j + 1) * batch_size
				m = int(50 * n / train_x.shape[0])
				bar = m * '=' + '>' + (49 - m) * ' '
				print('epoch {}/{} ({}/{}) [{}]'.format(i + 1 , epoch , n , train_x.shape[0] , bar) , end = '\r')
			else:
				n = train_x.shape[0]
				bar = 50 * '='
				end = time()
				print('epoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f} , train accuracy : {:.5f}'.format(i + 1 , epoch , n , train_x.shape[0] , bar , int(end - start) , total_loss / train_x.shape[0] , count / train_x.shape[0]))

		if ((i + 1) % 1 == 0):
			start = time()
			model.eval()
			count = 0
			total_loss = 0
			with torch.no_grad():
				for (data , label) in validation_loader:
					(data , label) = (data.to(device , dtype = torch.long) , label.to(device , dtype = torch.float))
					y = model(data)
					y = y.squeeze()
					index = (y >= 0.5).int()
					count += torch.sum(label == index).item()
					loss = criterion(y , label)
					total_loss += loss.item()
			end = time()
			print('evaluation ({}s) validation loss : {:.8f} ,  validation accuracy : {:.5f}'.format(int(end - start) , total_loss / validation_x.shape[0] , count / validation_x.shape[0]))

			if (count / validation_x.shape[0] > max_accuracy):
				max_accuracy = count / validation_x.shape[0]
				torch.save(model.state_dict() , 'RNN.pkl')

	return model

def generate_pseudolabel(train_x , train_y , nolabel_x , model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	threshold_1 = args.threshold_1
	threshold_2 = args.threshold_2

	nolabel_dataset = dataset(nolabel_x , None)
	nolabel_loader = DataLoader(nolabel_dataset , batch_size = batch_size , shuffle = False , num_workers = 8)

	start = time()
	model.to(device)
	model.eval()
	probability = list()
	predict = list()
	with torch.no_grad():
		for data in nolabel_loader:
			data = data.to(device , dtype = torch.long)
			y = model(data)
			y = y.squeeze()
			index = (y >= 0.5).float()
			probability.append(y.cpu().detach().numpy())
			predict.append(index.cpu().detach().numpy())
	probability = np.concatenate(probability , axis = 0)
	predict = np.concatenate(predict , axis = 0)
	index = [i for i in range(nolabel_x.shape[0]) if (probability[i] <= threshold_1 or probability[i] >= threshold_2)]
	train_x = np.concatenate((train_x , nolabel_x[index]) , axis = 0)
	train_y = np.concatenate((train_y , predict[index]) , axis = 0)
	nolabel_x = np.delete(nolabel_x , index , axis = 0)
	end = time()
	print('generate {} pseudo-label ({}s)'.format(len(index) , int(end - start)))

	return (train_x , train_y , nolabel_x , probability)

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(train_x , train_y , nolabel_x , validation_x , validation_y , vocabulary) = load_data(args)
	model = RNN(vocabulary.embedding , vocabulary.token2index['<PAD>'])
	model = train(train_x , train_y , validation_x , validation_y , model , device , args)
	(train_x , train_y , nolabel_x , probability) = generate_pseudolabel(train_x , train_y , nolabel_x , model , device , args)
	model = RNN(vocabulary.embedding , vocabulary.token2index['<PAD>'])
	model = train(train_x , train_y , validation_x , validation_y , model , device , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)
