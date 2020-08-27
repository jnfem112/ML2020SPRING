import argparse
import numpy as np
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms

class Dataset(Dataset):
	def __init__(self , image , label , transform):
		self.image = image
		self.label = label
		self.transform = transform
		return

	def __len__(self):
		return len(self.image)

	def __getitem__(self , index):
		return (self.transform(self.image[index]) , self.label[index])

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--input_directory' , type = str , default = 'data/')
	parser.add_argument('--output_directory' , type = str , default = 'output/')
	parser.add_argument('--method' , type = str , default = 'BIM')
	parser.add_argument('--epsilon' , type = float , default = 0.0135)
	args = parser.parse_args()
	return args

def get_transform():
	return transforms.Compose([
		transforms.ToTensor() ,
		transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])
	])

def get_dataloader(data , label):
	transform = get_transform()
	dataset = Dataset(data , label , transform)
	dataloader = DataLoader(dataset , batch_size = 1 , shuffle = False)
	return dataloader

def deprocess(image):
	image[0] = 0.229 * image[0] + 0.485
	image[1] = 0.224 * image[1] + 0.456
	image[2] = 0.225 * image[2] + 0.406
	image = 255 * image
	image = np.transpose(image , axes = (1 , 2 , 0))
	return image