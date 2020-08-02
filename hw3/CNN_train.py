import argparse
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from CNN import CNN
import torch.nn as nn
from torch.optim import Adam
from time import time

class dataset(Dataset):
	def __init__(self , data , label , transform):
		self.data = data
		self.label = label
		self.transform = transform
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		return (self.transform(self.data[index]) , self.label[index])

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--train_directory' , type = str)
	parser.add_argument('--validation_directory' , type = str)
	parser.add_argument('--batch_size' , type = int)
	parser.add_argument('--learning_rate' , type = float)
	parser.add_argument('--epoch' , type = int)
	args = parser.parse_args()
	return args

def load_data(args):
	train_x = list()
	train_y = list()
	for image_name in os.listdir(args.train_directory):
		image = cv2.imread(os.path.join(args.train_directory , image_name) , cv2.IMREAD_COLOR)
		image = cv2.resize(image , (128 , 128) , interpolation = cv2.INTER_CUBIC)
		label = int(image_name.split('_')[0])
		train_x.append(image)
		train_y.append(label)
	train_x = np.array(train_x)
	train_y = np.array(train_y)

	validation_x = list()
	validation_y = list()
	for image_name in os.listdir(args.validation_directory):
		image = cv2.imread(os.path.join(args.validation_directory , image_name) , cv2.IMREAD_COLOR)
		image = cv2.resize(image , (128 , 128) , interpolation = cv2.INTER_CUBIC)
		label = int(image_name.split('_')[0])
		validation_x.append(image)
		validation_y.append(label)
	validation_x = np.array(validation_x)
	validation_y = np.array(validation_y)

	return (train_x , train_y , validation_x , validation_y)

def train(train_x , train_y , validation_x , validation_y , model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	epoch = args.epoch

	train_transform = transforms.Compose([
		transforms.ToPILImage() ,
		transforms.RandomAffine(15 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
		transforms.RandomHorizontalFlip() ,
		transforms.ToTensor()
	])

	validation_transform = transforms.Compose([
		transforms.ToTensor()
	])

	train_dataset = dataset(train_x , train_y , train_transform)
	validation_dataset = dataset(validation_x , validation_y , validation_transform)
	train_loader = DataLoader(train_dataset , batch_size = batch_size , shuffle = True , num_workers = 8)
	validation_loader = DataLoader(validation_dataset , batch_size = batch_size , shuffle = False , num_workers = 8)

	model.to(device)
	optimizer = Adam(model.parameters() , lr = learning_rate)
	criterion = nn.CrossEntropyLoss()
	for i in range(epoch):
		start = time()
		model.train()
		count = 0
		total_loss = 0
		for (j , (data , label)) in enumerate(train_loader):
			(data , label) = (data.to(device) , label.to(device))
			optimizer.zero_grad()
			y = model(data)
			(_ , index) = torch.max(y , dim = 1)
			count += torch.sum(label == index).item()
			loss = criterion(y , label)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()

			if (j < len(train_loader) - 1):
				n = (j + 1) * batch_size
				m = int(50 * n / train_x.shape[0])
				bar = m * '=' + '>' + (49 - m) * ' '
				print('epoch {}/{} ({}/{}) [{}]'.format(i + 1 , epoch , n , train_x.shape[0] , bar) , end = '\r')
			else:
				n = train_x.shape[0]
				bar = 50 * '='
				end = time()
				print('epoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f} , train accuracy : {:.5f}'.format(i + 1 , epoch , n , train_x.shape[0] , bar , int(end - start) , total_loss / train_x.shape[0] , count / train_x.shape[0]))

		if ((i + 1) % 10 == 0):
			start = time()
			model.eval()
			count = 0
			total_loss = 0
			with torch.no_grad():
				for (data , label) in validation_loader:
					(data , label) = (data.to(device) , label.to(device))
					y = model(data)
					(_ , index) = torch.max(y , dim = 1)
					count += torch.sum(label == index).item()
					loss = criterion(y , label)
					total_loss += loss.item()
			end = time()
			print('evaluation ({}s) validation loss : {:.8f} ,  validation accuracy : {:.5f}'.format(int(end - start) , total_loss / validation_x.shape[0] , count / validation_x.shape[0]))

	return model

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(train_x , train_y , validation_x , validation_y) = load_data(args)
	model = CNN()
	model = train(train_x , train_y , validation_x , validation_y , model , device , args)
	torch.save(model.state_dict() , 'CNN.pkl')
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)