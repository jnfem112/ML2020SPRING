import argparse
import os
import random as rd
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader

class Dataset(Dataset):
	def __init__(self , directory , transform):
		self.directory = directory
		self.image_name = os.listdir(directory)
		self.transform = transform
		return

	def __len__(self):
		return len(self.image_name)

	def __getitem__(self , index):
		image = cv2.imread(os.path.join(self.directory , self.image_name[index]) , cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
		return self.transform(image)

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--train_directory' , type = str , default = 'faces/')
	parser.add_argument('--output_image' , type = str , default = 'output.jpg')
	parser.add_argument('--checkpoint' , type = str , default = 'DCGAN_generator.pkl')
	parser.add_argument('--model' , type = str , default = 'DCGAN')
	parser.add_argument('--input_dim' , type = int , default = 100)
	parser.add_argument('--batch_size' , type = int , default = 64)
	parser.add_argument('--learning_rate' , type = float , default = 0.0001)
	parser.add_argument('--clip_value' , type = float , default = 0.01)
	parser.add_argument('--lambd' , type = float , default = 10)
	parser.add_argument('--n_critic' , type = int , default = 5)
	parser.add_argument('--epoch' , type = int , default = 10)
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

def get_transform():
	return transforms.Compose([
		transforms.ToPILImage() ,
		transforms.Resize((64 , 64)) ,
		transforms.ToTensor() ,
		transforms.Normalize(mean = [0.5 , 0.5 , 0.5] , std = [0.5 , 0.5 , 0.5])
	])

def get_dataloader(directory , batch_size = 1024 , num_workers = 8):
	transform = get_transform()
	dataset = Dataset(directory , transform)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = True , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss_generator = None , loss_discriminator = None):
	if (batch < total_batch):
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('\repoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '')
	else:
		data = total_data
		bar = 50 * '='
		print('\repoch {}/{} ({}/{}) [{}] ({}s) generator loss : {:.8f} , discriminator loss : {:.8f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss_generator , loss_discriminator))
	return