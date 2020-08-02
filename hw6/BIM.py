import argparse
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset , DataLoader
from torchvision import models , transforms
import torch.nn as nn
from time import time

class dataset(Dataset):
	def __init__(self , original_image , original_label , transform):
		self.original_image = original_image
		self.original_label = original_label
		self.transform = transform
		return

	def __len__(self):
		return len(self.original_image)

	def __getitem__(self , index):
		return (self.transform(self.original_image[index]) , self.original_label[index])

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--input_directory' , type = str)
	parser.add_argument('--output_directory' , type = str)
	parser.add_argument('--epsilon' , type = int)
	args = parser.parse_args()
	return args

def load_data(args):
	original_image = list()
	original_label = list()
	df = pd.read_csv(os.path.join(args.input_directory , 'labels.csv'))
	for i in range(df.shape[0]):
		original_image.append(Image.open(os.path.join(args.input_directory , 'images/{:03d}.png'.format(df['ImgId'][i]))))
		original_label.append(df['TrueLabel'][i])
	return (original_image , original_label)

def BIM(original_image , original_label , proxy_model , device , args):
	# Hyper-parameter.
	epsilon = args.epsilon

	transform = transforms.Compose([
		transforms.ToTensor() ,
		transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])
	])

	attack_dataset = dataset(original_image , original_label , transform)
	attack_dataloader = DataLoader(attack_dataset , batch_size = 1 , shuffle = False)

	proxy_model.to(device)
	proxy_model.eval()
	criterion = nn.CrossEntropyLoss()
	new_image = list()
	new_label = list()
	for (i , (image , label)) in enumerate(attack_dataloader):
		print('image {} '.format(i + 1) , end = '')
		start = time()
		(image , label) = (image.to(device) , label.to(device))
		while True:
			image.requires_grad = True
			y = proxy_model(image)
			(_ , index) = torch.max(y , dim = 1)
			loss = criterion(y , label)
			loss.backward()
			if (index[0].item() != label.item()):
				break
			else:
				image = nn.Parameter(image + epsilon * torch.sign(image.grad))
		end = time()
		print('({}s) '.format(int(end - start)))

		image = image.cpu().detach().squeeze(dim = 0).numpy()
		image = deprocess(image)
		new_image.append(image)
		new_label.append(index[0].item())

	return (new_image , new_label)

def deprocess(image):
	image[0] = 0.229 * image[0] + 0.485
	image[1] = 0.224 * image[1] + 0.456
	image[2] = 0.225 * image[2] + 0.406
	image = 255 * image
	image = np.transpose(image , axes = (1 , 2 , 0))
	return image

def save_image(new_image , args):
	for (i , image) in enumerate(new_image):
		image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)
		cv2.imwrite(os.path.join(args.output_directory , '{:03d}.png'.format(i)) , image)
	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(original_image , original_label) = load_data(args)
	proxy_model = models.densenet121(pretrained = True)
	(new_image , new_label) = BIM(original_image , original_label , proxy_model , device , args)
	save_image(new_image , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)
