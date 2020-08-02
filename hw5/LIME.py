import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from CNN import CNN
from skimage.segmentation import slic
from lime import lime_image
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

def classifier_fn(image):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	transform = transforms.ToTensor()
	image = torch.stack([transform(image[i]) for i in range(image.shape[0])])
	image = image.to(device)
	model = CNN()
	model.load_state_dict(torch.load('CNN.pkl' , map_location = device))
	model.to(device)
	model.eval()
	y = model(image)
	y = y.cpu().detach().numpy()
	return y

def segmentation_fn(image):
	return slic(image , n_segments = 100 , compactness = 1 , sigma = 1)

def LIME(image , label):
	explainer = lime_image.LimeImageExplainer()
	explaination = explainer.explain_instance(image = image , classifier_fn = classifier_fn , segmentation_fn = segmentation_fn)
	(result , mask) = explaination.get_image_and_mask(label = label , num_features = 11 , positive_only = False , hide_rest = False , min_weight = 0.05)
	return result

def plot(image , result , args):
	image = image[... , : : -1]
	result = result[... , : : -1]

	fig = plt.figure(figsize = (12 , 6))

	ax1 = fig.add_subplot(1 , 2 , 1)
	ax1.set_title('Original Image' , fontsize = 24 , y = 1.02)
	ax1.imshow(image)
	ax1.set_axis_off()

	ax2 = fig.add_subplot(1 , 2 , 2)
	ax2.set_title('LIME' , fontsize = 24 , y = 1.02)
	ax2.imshow(result)
	ax2.set_axis_off()

	plt.savefig(os.path.join(args.output_directory , args.output_file))

	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	(image , label) = load_data(args)
	result = LIME(image , label)
	plot(image , result , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)