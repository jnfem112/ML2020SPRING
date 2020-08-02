import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms , models
from torch.optim import Adam
from CNN import CNN
import matplotlib.pyplot as plt

activation = None

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--image_directory' , type = str)
	parser.add_argument('--image_name' , type = str)
	parser.add_argument('--layer' , type = int)
	parser.add_argument('--learning_rate' , type = float)
	parser.add_argument('--epoch' , type = int)
	parser.add_argument('--output_directory' , type = str)
	parser.add_argument('--output_file' , type = str)
	args = parser.parse_args()
	return args

def load_data(args):
	image = cv2.imread(os.path.join(args.image_directory , args.image_name) , cv2.IMREAD_COLOR)
	image = cv2.resize(image , (128 , 128) , interpolation = cv2.INTER_CUBIC)
	return image

def deep_dream(image , model , device , args):
	# Hyper-parameter.
	learning_rate = args.learning_rate
	epoch = args.epoch

	def hook(model , input , output):
		global activation
		activation = output
		return

	global activation

	transform = transforms.ToTensor()
	dream = np.copy(image)
	dream = transform(dream).unsqueeze(dim = 0)
	dream = dream.to(device)
	dream.requires_grad = True
	model.to(device)
	model.eval()
	hook_handle = model.convolution[args.layer].register_forward_hook(hook)
	optimizer = Adam([dream] , lr = learning_rate)
	for i in range(epoch):
		optimizer.zero_grad()
		model(dream)
		loss = -torch.norm(activation)
		loss.backward(retain_graph = True)
		optimizer.step()
	hook_handle.remove()

	dream = dream.cpu().detach().squeeze(dim = 0).numpy()
	dream = np.transpose(dream , axes = (1 , 2 , 0))
	dream = (dream - np.min(dream)) / (np.max(dream) - np.min(dream))

	return dream

def plot(image , dream , args):
	image = image[... , : : -1]
	dream = dream[... , : : -1]

	fig = plt.figure(figsize = (12 , 6))

	ax1 = fig.add_subplot(1 , 2 , 1)
	ax1.set_title('Original Image' , fontsize = 24 , y = 1.02)
	ax1.imshow(image)
	ax1.set_axis_off()

	ax2 = fig.add_subplot(1 , 2 , 2)
	ax2.set_title('Deep Dream' , fontsize = 24 , y = 1.02)
	ax2.imshow(dream)
	ax2.set_axis_off()

	plt.savefig(os.path.join(args.output_directory , args.output_file))

	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	image = load_data(args)
	model = CNN()
	model.load_state_dict(torch.load('CNN.pkl' , map_location = device))
	dream = deep_dream(image , model , device , args)
	plot(image , dream , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)