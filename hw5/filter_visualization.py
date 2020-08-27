import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from model import CNN
from torch.optim import Adam
import matplotlib.pyplot as plt

activation = None

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--layer' , type = int)
	parser.add_argument('--filter' , type = int)
	parser.add_argument('--learning_rate' , type = float , default = 1)
	parser.add_argument('--epoch' , type = int , default = 200)
	parser.add_argument('--output_directory' , type = str)
	parser.add_argument('--output_file' , type = str)
	args = parser.parse_args()
	return args

def filter_visualization(model , device , args):
	# Hyper-parameter.
	learning_rate = args.learning_rate
	epoch = args.epoch

	def hook(model , input , output):
		global activation
		activation = output
		return

	global activation

	model.to(device)
	model.eval()
	hook_handle = model.convolution[args.layer].register_forward_hook(hook)
	feature = torch.rand((1 , 3 , 128 , 128) , device = device , requires_grad = True)
	optimizer = Adam([feature] , lr = learning_rate)
	for i in range(epoch):
		optimizer.zero_grad()
		model(feature)
		objective = -torch.sum(activation[ : , args.filter , : , : ])
		objective.backward()
		optimizer.step()
	hook_handle.remove()
 
	feature = feature.cpu().detach().squeeze(dim = 0).numpy()
	feature = np.transpose(feature , axes = (1 , 2 , 0))
	feature = feature[... , : : -1]
	feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
 
	return feature

def plot(feature , args):
	plt.title('layer {}, filter {}'.format(args.layer , args.filter) , fontsize = 20 , y = 1.02)
	plt.imshow(feature)
	plt.axis('off')
	plt.savefig(os.path.join(args.output_directory , args.output_file))
	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = CNN()
	model.load_state_dict(torch.load('CNN.pkl' , map_location = device))
	feature = filter_visualization(model , device , args)
	plot(feature , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)