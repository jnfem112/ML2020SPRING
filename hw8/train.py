import os
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from time import time
from utils import my_argparse , get_dataloader , BLEU , print_progress
from data import load_dataset , deprocess
from model import Encoder , Decoder , Seq2Seq

def scheduled_sampling(step , base = 1 , decay = 0.025 , threshold = 0 , method = 'constant'):
	if (method == 'linear'):
		return max(threshold , base - decay * step)
	elif (method == 'exponential'):
		return math.pow(base , step)
	elif (method == 'sigmoid'):
		return base / (base + math.exp(step / base))
	else:
		return base

def train(train_x , train_y , validation_x , validation_y , dictionary , model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	epoch = args.epoch

	train_loader = get_dataloader(train_x , train_y , 'train' , batch_size)
	model.to(device)
	optimizer = Adam(model.parameters() , lr = learning_rate)
	criterion = nn.CrossEntropyLoss(ignore_index = dictionary.token2index_chinese['<PAD>'])
	max_score = 0
	for i in range(epoch):
		model.train()
		total_loss = 0
		start = time()
		for (j , (input , target)) in enumerate(train_loader):
			(input , target) = (input.to(device) , target.to(device))
			optimizer.zero_grad()
			teacher_forcing_ratio = scheduled_sampling(i , base = 1 , decay = 1 / epoch , threshold = 0 , method = 'linear')
			(output , predict) = model(input , target , teacher_forcing_ratio)
			output = output.reshape(-1 , output.size(dim = 2))
			target = target[ : , 1 : ].reshape(-1)
			loss = criterion(output , target)
			total_loss += loss.item()
			grad_norm = clip_grad_norm_(model.parameters() , max_norm = 1)
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , epoch , train_x.shape[0] , batch_size , j + 1 , len(train_loader) , int(end - start) , total_loss / train_x.shape[0])

		if ((i + 1) % 5 == 0):
			score = evaluate(validation_x , validation_y , dictionary , model , device , args)
			if (score > max_score):
				max_score = score
				torch.save(model.state_dict() , 'Seq2Seq.pkl')

	return model

def evaluate(validation_x , validation_y , dictionary , model , device , args):
	validation_loader = get_dataloader(validation_x , validation_y , 'validation' , batch_size = 1)
	model.to(device)
	model.eval()
	criterion = nn.CrossEntropyLoss(ignore_index = dictionary.token2index_chinese['<PAD>'])
	total_loss = 0
	total_score = 0
	start = time()
	with torch.no_grad():
		for (input , target) in validation_loader:
			(input , target) = (input.to(device) , target.to(device))
			(output , predict) = model.inference(input , target , args.beam_size)
			output = output.reshape(-1 , output.size(dim = 2))
			target = target[ : , 1 : ].reshape(-1)
			loss = criterion(output , target)
			total_loss += loss.item()
			(predict , target) = (predict.squeeze(dim = 0) , target.squeeze(dim = 0))
			predict = deprocess(predict , dictionary.index2token_chinese)
			target = deprocess(target , dictionary.index2token_chinese)
			total_score += BLEU(predict , target)
	end = time()
	print('evaluation ({}s) validation loss : {:.8f} , validation BLEU : {:.5f}'.format(int(end - start) , total_loss / validation_x.shape[0] , total_score / validation_x.shape[0]))
	return total_score / validation_x.shape[0]

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(train_x , train_y , validation_x , validation_y , dictionary) = load_dataset(args.root , 'train' , args.max_length)
	encoder = Encoder(len(dictionary.token2index_english) , embedding_dim = 256 , hidden_dim = 512 , num_layers = 3 , dropout = 0.5)
	decoder = Decoder(len(dictionary.token2index_chinese) , embedding_dim = 256 , hidden_dim = 1024 , num_layers = 3 , dropout = 0.5)
	model = Seq2Seq(encoder , decoder)
	model = train(train_x , train_y , validation_x , validation_y , dictionary , model , device , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)