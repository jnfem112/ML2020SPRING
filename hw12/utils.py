import argparse
import os
import numpy as np
import cv2
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--directory' , type = str , default = 'real_or_drawing/')
	parser.add_argument('--output_file' , type = str , default = 'predict.csv')
	parser.add_argument('--batch_size' , type = int , default = 128)
	parser.add_argument('--learning_rate' , type = float , default = 0.00002)
	parser.add_argument('--weight_decay' , type = float , default = 0.0005)
	parser.add_argument('--epoch' , type = int , default = 2000)
	args = parser.parse_args()
	return args

def get_transform(mode):
	if (mode == 'train'):
		source_transform = transforms.Compose([
			transforms.Grayscale() ,
			transforms.Lambda(lambda x : cv2.Canny(np.array(x) , 250 , 300)) ,
			transforms.ToPILImage() ,
			transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
			transforms.RandomHorizontalFlip() ,
			transforms.ToTensor()
		])

		target_transform = transforms.Compose([
			transforms.Grayscale() ,
			transforms.Resize((32 , 32)) ,
			transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
			transforms.RandomHorizontalFlip() ,
			transforms.ToTensor()
		])

		return (source_transform , target_transform)
	else:
		target_transform = transforms.Compose([
			transforms.Grayscale() ,
			transforms.Resize((32 , 32)) ,
			transforms.ToTensor()
		])
		
		return target_transform

def get_dataloader(directory , mode , batch_size = 1024 , num_workers = 8):
	if (mode == 'train'):
		(source_transform , target_transform) = get_transform(mode)
		source_dataset = ImageFolder(os.path.join(directory , 'train_data/') , transform = source_transform)
		target_dataset = ImageFolder(os.path.join(directory , 'test_data/') , transform = target_transform)
		source_dataloader = DataLoader(source_dataset , batch_size = batch_size , shuffle = True , num_workers = num_workers)
		target_dataloader = DataLoader(target_dataset , batch_size = batch_size , shuffle = True , num_workers = num_workers)
		return (source_dataloader , target_dataloader)
	else:
		target_transform = get_transform(mode)
		target_dataset = ImageFolder(os.path.join(directory , 'test_data/') , transform = target_transform)
		target_dataloader = DataLoader(target_dataset , batch_size = batch_size , shuffle = False , num_workers = num_workers)
		return target_dataloader

def print_progress(epoch , total_epoch , batch , total_batch , total_time = None):
	if (batch < total_batch):
		length = int(50 * batch / total_batch)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} [{}]'.format(epoch , total_epoch , bar) , end = '')
	else:
		bar = 50 * '='
		print('\repoch {}/{} [{}] ({}s)'.format(epoch , total_epoch , bar , total_time))
	return