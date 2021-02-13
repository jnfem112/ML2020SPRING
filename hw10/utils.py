import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--train_data' , type = str , default = 'train.npy')
	parser.add_argument('--test_data' , type = str , default = 'test.npy')
	parser.add_argument('--output_file' , type = str , default = 'predict.csv')
	parser.add_argument('--batch_size' , type = int , default = 128)
	parser.add_argument('--learning_rate' , type = float , default = 0.001)
	parser.add_argument('--epoch' , type = int , default = 50)
	parser.add_argument('--n_clusters' , type = int , default = 5)
	args = parser.parse_args()
	return args

def get_dataloader(data , mode , batch_size = 1024 , num_workers = 8):
	data = np.transpose(data , axes = (0 , 3 , 1 , 2))
	data = torch.FloatTensor(data)
	dataloader = DataLoader(data , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss = None):
	if (batch < total_batch):
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '')
	else:
		data = total_data
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss))
	return