import os
import torch
import torch.nn as nn
from torch.optim import Adam
from time import time
from utils import my_argparse , set_random_seed , get_dataloader , print_progress
from data import load_data
from model import Autoencoder

def train(dataset , model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	weight_decay = args.weight_decay
	epoch = args.epoch

	set_random_seed(0)

	dataloader = get_dataloader(dataset , 'train' , args.data_augmentation , batch_size)
	model.to(device)
	optimizer = Adam(model.parameters() , lr = learning_rate , weight_decay = weight_decay)
	criterion = nn.MSELoss()
	for i in range(epoch):
		model.train()
		total_loss = 0
		start = time()
		for (j , data) in enumerate(dataloader):
			data = data.to(device)
			optimizer.zero_grad()
			(encode , decode) = model(data)
			loss = criterion(data , decode)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , epoch , dataset.shape[0] , batch_size , j + 1 , len(dataloader) , int(end - start) , total_loss / dataset.shape[0])

	return model

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_x = load_data(args.train_x)
	model = Autoencoder()
	model = train(train_x , model , device , args)
	torch.save(model.state_dict() , args.checkpoint)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)