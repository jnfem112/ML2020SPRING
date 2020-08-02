import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from CNN import CNN
import matplotlib.pyplot as plt

activation = None

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--image_directory' , type = str)
	parser.add_argument('--image_name' , type = str)
	parser.add_argument('--layer' , type = int)
	parser.add_argument('--filter' , type = int)
	parser.add_argument('--output_directory' , type = str)
	parser.add_argument('--output_file' , type = str)
	args = parser.parse_args()
	return args

def load_data(args):
	image = cv2.imread(os.path.join(args.image_directory , args.image_name) , cv2.IMREAD_COLOR)
	image = cv2.resize(image , (128 , 128) , interpolation = cv2.INTER_CUBIC)
	return image

def activation_map(image , model , device , args):
	def hook(model , input , output):
		global activation
		activation = output
		return

	global activation

	transform = transforms.ToTensor()
	image = transform(image).unsqueeze(dim = 0)
	image = image.to(device)
	model.to(device)
	model.eval()
	hook_handle = model.convolution[args.layer].register_forward_hook(hook)

	model(image)
	activation = activation[ : , args.filter , : , : ]
	hook_handle.remove()

	activation = activation.cpu().detach().squeeze(dim = 0).numpy()
	activation = (activation - np.min(activation)) / (np.max(activation) - np.min(activation))

	return activation

def plot(image , activation , args):
	image = image[... , : : -1]
	activation = cv2.resize(activation , (128 , 128) , interpolation = cv2.INTER_CUBIC)

	fig = plt.figure(figsize = (18 , 6))

	ax1 = fig.add_subplot(1 , 3 , 1)
	ax1.set_title('Original Image' , fontsize = 24 , y = 1.02)
	ax1.imshow(image)
	ax1.set_axis_off()

	ax2 = fig.add_subplot(1 , 3 , 2)
	ax2.set_title('Activation Map\n(layer {} , filter {})'.format(args.layer , args.filter) , fontsize = 24 , y = 1.02)
	ax2.imshow(activation , cmap = 'jet')
	ax2.set_axis_off()

	ax3 = fig.add_subplot(1 , 3 , 3)
	ax3.set_title('Overlay' , fontsize = 24 , y = 1.02)
	ax3.imshow(cv2.cvtColor(image , cv2.COLOR_BGR2GRAY) , cmap = 'gray')
	ax3.imshow(activation , cmap = 'jet' , alpha = 0.5)
	ax3.set_axis_off()

	plt.savefig(os.path.join(args.output_directory , args.output_file))

	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	image = load_data(args)
	model = CNN()
	model.load_state_dict(torch.load('CNN.pkl' , map_location = device))
	activation = activation_map(image , model , device , args)
	plot(image , activation , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)