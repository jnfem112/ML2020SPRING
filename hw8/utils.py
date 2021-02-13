import argparse
import os
import json
import torch
from torch.utils.data import Dataset , DataLoader
from nltk.translate.bleu_score import sentence_bleu , SmoothingFunction

class Dictionary():
	def __init__(self , root):
		with open(os.path.join(root , 'word2int_en.json') , 'r') as file:
			self.token2index_english = json.load(file)
		with open(os.path.join(root , 'int2word_en.json') , 'r') as file:
			self.index2token_english = json.load(file)
		with open(os.path.join(root , 'word2int_cn.json') , 'r') as file:
			self.token2index_chinese = json.load(file)
		with open(os.path.join(root , 'int2word_cn.json') , 'r') as file:
			self.index2token_chinese = json.load(file)
		return

class Dataset(Dataset):
	def __init__(self , input , target):
		self.input = input
		self.target = target
		return

	def __len__(self):
		return self.input.shape[0]

	def __getitem__(self , index):
		return (torch.LongTensor(self.input[index]) , torch.LongTensor(self.target[index]))

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--root' , type = str , default = 'cmn-eng/')
	parser.add_argument('--output_file' , type = str , default = 'output.txt')
	parser.add_argument('--max_length' , type = int , default = 50)
	parser.add_argument('--batch_size' , type = int , default = 64)
	parser.add_argument('--learning_rate' , type = float , default = 0.0001)
	parser.add_argument('--epoch' , type = int , default = 40)
	parser.add_argument('--beam_size' , type = int , default = 4)
	args = parser.parse_args()
	return args

def get_dataloader(input , target , mode , batch_size = 1024 , num_workers = 8):
	dataset = Dataset(input , target)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

def BLEU(predict , target):
	def cut_token(sentence):
		result = list()
		for token in sentence:
			if (token == '<UNK>' or token.isdigit() or len(bytes(token[0] , encoding = 'utf-8')) == 1):
				result.append(token)
			else:
				result += [word for word in token]
		return result 

	predict = cut_token(predict)
	target = cut_token(target)
	score = sentence_bleu([target] , predict , weights = (1 , 0 , 0 , 0) , smoothing_function = SmoothingFunction().method4)
	return score

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