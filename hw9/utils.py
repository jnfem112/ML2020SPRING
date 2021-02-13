import argparse
import random as rd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader

class Dataset(Dataset):
	def __init__(self , data , transform):
		self.data = data
		self.transform = transform
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		data = self.data[index]
		data = self.transform(data)
		data = 2 * data - 1
		return data

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--train_x' , type = str , default = 'trainX.npy')
	parser.add_argument('--validation_x' , type = str , default = 'valX.npy')
	parser.add_argument('--validation_y' , type = str , default = 'valY.npy')
	parser.add_argument('--output_file' , type = str , default = 'predict.csv')
	parser.add_argument('--checkpoint' , type = str , default = 'Autoencoder.pth')
	parser.add_argument('--data_augmentation' , type = int , default = 1)
	parser.add_argument('--batch_size' , type = int , default = 16)
	parser.add_argument('--learning_rate' , type = float , default = 0.0001)
	parser.add_argument('--weight_decay' , type = float , default = 0.00001)
	parser.add_argument('--epoch' , type = int , default = 100)
	args = parser.parse_args()
	return args

def set_random_seed(seed):
	rd.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if (torch.cuda.is_available()):
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	return

def get_transform(mode , data_augmentation):
	if (mode == 'train' and data_augmentation):
		return transforms.Compose([
			transforms.ToPILImage() ,
			transforms.RandomAffine(15 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
			transforms.RandomHorizontalFlip() ,
			transforms.ColorJitter(brightness = 0.1 , contrast = 0.1 , saturation = 0.1 , hue = 0.1) ,
			transforms.ToTensor()
		])
	else:
		return transforms.Compose([
			transforms.ToTensor()
		])

def get_dataloader(data , mode , data_augmentation = False , batch_size = 1024 , num_workers = 8):
	transform = get_transform(mode , data_augmentation)
	dataset = Dataset(data , transform)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss = None):
	if (batch < total_batch):
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '')
	else:
		data = total_data
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss))
	return