import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from time import time
from utils import my_argparse , get_dataloader , print_progress
from data import load_data
from model import RNN

max_accuracy = 0

def train(train_x , train_y , validation_x , validation_y , model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	epoch = args.epoch

	train_dataloader = get_dataloader(train_x , train_y , 'train' , batch_size)
	model.to(device)
	optimizer = Adam(model.parameters() , lr = learning_rate)
	criterion = nn.BCELoss()
	for i in range(epoch):
		model.train()
		count = 0
		total_loss = 0
		start = time()
		for (j , (data , label)) in enumerate(train_dataloader):
			(data , label) = (data.to(device , dtype = torch.long) , label.to(device , dtype = torch.float))
			optimizer.zero_grad()
			output = model(data)
			output = output.squeeze()
			index = (output >= 0.5).int()
			count += torch.sum(label == index).item()
			loss = criterion(output , label)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , epoch , train_x.shape[0] , batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss / train_x.shape[0] , count / train_x.shape[0])

		if ((i + 1) % 1 == 0):
			evaluate(validation_x , validation_y , model , device)

	return model

def evaluate(validation_x , validation_y , model , device):
	global max_accuracy

	validation_dataloader = get_dataloader(validation_x , validation_y , 'validation')
	model.to(device)
	model.eval()
	criterion = nn.BCELoss()
	count = 0
	total_loss = 0
	start = time()
	with torch.no_grad():
		for (data , label) in validation_dataloader:
			(data , label) = (data.to(device , dtype = torch.long) , label.to(device , dtype = torch.float))
			output = model(data)
			output = output.squeeze()
			index = (output >= 0.5).int()
			count += torch.sum(label == index).item()
			loss = criterion(output , label)
			total_loss += loss.item()
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / validation_x.shape[0] , count / validation_x.shape[0]))
	
	if (count / validation_x.shape[0] > max_accuracy):
		max_accuracy = count / validation_x.shape[0]
		torch.save(model.state_dict() , 'RNN.pkl')

	return

def generate_pseudolabel(train_x , train_y , nolabel_x , model , device , args):
	# Hyper-parameter.
	threshold_1 = args.threshold_1
	threshold_2 = args.threshold_2

	nolabel_loader = get_dataloader(nolabel_x , None , 'nolabel')
	model.to(device)
	model.eval()
	probability = list()
	predict = list()
	start = time()
	with torch.no_grad():
		for data in nolabel_loader:
			data = data.to(device , dtype = torch.long)
			output = model(data)
			output = output.squeeze()
			index = (output >= 0.5).float()
			probability.append(output.cpu().detach().numpy())
			predict.append(index.cpu().detach().numpy())
	probability = np.concatenate(probability , axis = 0)
	predict = np.concatenate(predict , axis = 0)
	index = [i for i in range(nolabel_x.shape[0]) if (probability[i] <= threshold_1 or probability[i] >= threshold_2)]
	train_x = np.concatenate((train_x , nolabel_x[index]) , axis = 0)
	train_y = np.concatenate((train_y , predict[index]) , axis = 0)
	nolabel_x = np.delete(nolabel_x , index , axis = 0)
	end = time()
	print('generate {} pseudo-label ({}s)'.format(len(index) , int(end - start)))

	return (train_x , train_y , nolabel_x)

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(train_x , train_y , nolabel_x , validation_x , validation_y , vocabulary) = load_data(args.train_data , args.nolabel_data , None , args.max_length)
	model = RNN(vocabulary.embedding , vocabulary.token2index['<PAD>'])
	model = train(train_x , train_y , validation_x , validation_y , model , device , args)
	(train_x , train_y , nolabel_x) = generate_pseudolabel(train_x , train_y , nolabel_x , model , device , args)
	model = RNN(vocabulary.embedding , vocabulary.token2index['<PAD>'])
	model = train(train_x , train_y , validation_x , validation_y , model , device , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)