import argparse
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader
from gensim.models import Word2Vec

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

class Dataset(Dataset):
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
	parser.add_argument('--train_data' , type = str , default = 'training_label.txt')
	parser.add_argument('--nolabel_data' , type = str , default = 'training_nolabel.txt')
	parser.add_argument('--test_data' , type = str , default = 'testing_data.txt')
	parser.add_argument('--output_file' , type = str , default = 'predict.csv')
	parser.add_argument('--embedding_dimension' , type = int , default = 256)
	parser.add_argument('--iteration' , type = int , default = 10)
	parser.add_argument('--max_length' , type = int , default = 32)
	parser.add_argument('--batch_size' , type = int , default = 32)
	parser.add_argument('--learning_rate' , type = float , default = 0.001)
	parser.add_argument('--epoch' , type = int , default = 5)
	parser.add_argument('--threshold_1' , type = float , default = 0.1)
	parser.add_argument('--threshold_2' , type = float , default = 0.9)
	args = parser.parse_args()
	return args

def get_dataloader(data , label , mode , batch_size = 1024 , num_workers = 8):
	dataset = Dataset(data , label)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss = None , accuracy = None):
	if (batch < total_batch):
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('epoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '\r')
	else:
		data = total_data
		bar = 50 * '='
		print('epoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f} , train accuracy : {:.5f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss , accuracy))
	return