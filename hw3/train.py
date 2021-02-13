import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from time import time
from utils import my_argparse , get_dataloader , print_progress
from data import load_dataset
from model import CNN

def train(train_x , train_y , validation_x , validation_y , model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	epoch = args.epoch

	train_dataloader = get_dataloader(train_x , train_y , 'train' , batch_size)
	model.to(device)
	optimizer = Adam(model.parameters() , lr = learning_rate)
	criterion = nn.CrossEntropyLoss()
	for i in range(epoch):
		model.train()
		count = 0
		total_loss = 0
		start = time()
		for (j , (data , label)) in enumerate(train_dataloader):
			(data , label) = (data.to(device) , label.to(device))
			optimizer.zero_grad()
			output = model(data)
			(_ , index) = torch.max(output , dim = 1)
			count += torch.sum(label == index).item()
			loss = criterion(output , label)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , epoch , train_x.shape[0] , batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss / train_x.shape[0] , count / train_x.shape[0])

		if ((i + 1) % 10 == 0):
			evaluate(validation_x , validation_y , model , device)

	return model

def evaluate(validation_x , validation_y , model , device):
	validation_dataloader = get_dataloader(validation_x , validation_y , 'validation')
	model.to(device)
	model.eval()
	criterion = nn.CrossEntropyLoss()
	count = 0
	total_loss = 0
	start = time()
	with torch.no_grad():
		for (data , label) in validation_dataloader:
			(data , label) = (data.to(device) , label.to(device))
			output = model(data)
			(_ , index) = torch.max(output , dim = 1)
			count += torch.sum(label == index).item()
			loss = criterion(output , label)
			total_loss += loss.item()
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / validation_x.shape[0] , count / validation_x.shape[0]))
	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(train_x , train_y , validation_x , validation_y) = load_dataset(args.train_directory , args.validation_directory , None)
	model = CNN()
	model = train(train_x , train_y , validation_x , validation_y , model , device , args)
	torch.save(model.state_dict() , 'CNN.pkl')
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)