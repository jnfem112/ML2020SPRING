import argparse
import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from CNN import CNN

class dataset(Dataset):
	def __init__(self , data , transform):
		self.data = data
		self.transform = transform
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		return self.transform(self.data[index])

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--test_directory' , type = str)
	parser.add_argument('--output_file' , type = str)
	parser.add_argument('--batch_size' , type = int)
	args = parser.parse_args()
	return args

def load_data(args):
	test_x = list()
	for image_name in sorted(os.listdir(args.test_directory)):
		image = cv2.imread(os.path.join(args.test_directory , image_name) , cv2.IMREAD_COLOR)
		image = cv2.resize(image , (128 , 128) , interpolation = cv2.INTER_CUBIC)
		test_x.append(image)
	test_x = np.array(test_x)
	return test_x

def test(test_x , model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size

	test_transform = transforms.Compose([
		transforms.ToTensor()
	])

	test_dataset = dataset(test_x , test_transform)
	test_loader = DataLoader(test_dataset , batch_size = batch_size , shuffle = False , num_workers = 8)

	model.to(device)
	model.eval()
	test_y = list()
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)
			y = model(data)
			(_ , index) = torch.max(y , dim = 1)
			test_y.append(index.cpu().detach().numpy())
	test_y = np.concatenate(test_y , axis = 0)

	return test_y

def dump(test_y , args):
	number_of_data = test_y.shape[0]
	df = pd.DataFrame({'Id' : np.arange(number_of_data) , 'Category' : test_y})
	df.to_csv(args.output_file , index = False)
	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_x = load_data(args)
	model = CNN()
	model.load_state_dict(torch.load('CNN.pkl' , map_location = device))
	test_y = test(test_x , model , device , args)
	dump(test_y , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)