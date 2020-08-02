import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from CNN import CNN
import matplotlib.pyplot as plt

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--image_directory' , type = str)
	parser.add_argument('--image_name' , type = str)
	parser.add_argument('--output_directory' , type = str)
	parser.add_argument('--output_file' , type = str)
	args = parser.parse_args()
	return args

def load_data(args):
	image = cv2.imread(os.path.join(args.image_directory , args.image_name) , cv2.IMREAD_COLOR)
	image = cv2.resize(image , (128 , 128) , interpolation = cv2.INTER_CUBIC)
	label = int(image_name.split('_')[0])
	return (image , label)

def saliency_map(image , label , model , device):
	transform = transforms.ToTensor()
	(image , label) = (transform(image).unsqueeze(dim = 0) , torch.LongTensor([label]))
	(image , label) = (image.to(device) , label.to(device))
	image.requires_grad = True
	model.to(device)
	model.eval()
	criterion = nn.CrossEntropyLoss()

	y = model(image)
	loss = criterion(y , label)
	loss.backward()
	saliency = image.grad.abs()

	saliency = saliency.cpu().detach().squeeze(dim = 0).numpy()
	saliency = np.transpose(saliency , axes = (1 , 2 , 0))
	saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))

	return saliency

def plot(image , saliency , args):
	image = image[... , : : -1]
	saliency = saliency[... , : : -1]

	fig = plt.figure(figsize = (18 , 6))

	ax1 = fig.add_subplot(1 , 3 , 1)
	ax1.set_title('Original Image' , fontsize = 24 , y = 1.02)
	ax1.imshow(image)
	ax1.set_axis_off()

	ax2 = fig.add_subplot(1 , 3 , 2)
	ax2.set_title('Saliency Map' , fontsize = 24 , y = 1.02)
	ax2.imshow(saliency)
	ax2.set_axis_off()

	ax3 = fig.add_subplot(1 , 3 , 3)
	ax3.set_title('Overlay' , fontsize = 24 , y = 1.02)
	ax3.imshow(cv2.cvtColor(image , cv2.COLOR_BGR2GRAY) , cmap = 'gray')
	ax3.imshow(cv2.cvtColor(saliency , cv2.COLOR_BGR2GRAY) , cmap = 'jet' , alpha = 0.5)
	ax3.set_axis_off()

	plt.savefig(os.path.join(args.output_directory , args.output_file))

	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(image , label) = load_data(args)
	model = CNN()
	model.load_state_dict(torch.load('CNN.pkl' , map_location = device))
	saliency = saliency_map(image , label , model , device)
	plot(image , saliency , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)