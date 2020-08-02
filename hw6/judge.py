import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset , DataLoader
from torchvision import models , transforms

class dataset(Dataset):
	def __init__(self , new_image , original_label , transform):
		self.new_image = new_image
		self.original_label = original_label
		self.transform = transform
		return

	def __len__(self):
		return len(self.new_image)

	def __getitem__(self , index):
		return (self.transform(self.new_image[index]) , self.original_label[index])

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--input_directory' , type = str)
	parser.add_argument('--output_directory' , type = str)
	args = parser.parse_args()
	return args

def load_data(args):
	original_image = list()
	new_image = list()
	original_label = list()
	df = pd.read_csv(os.path.join(args.input_directory , 'labels.csv'))
	for i in range(df.shape[0]):
		original_image.append(Image.open(os.path.join(args.input_directory , 'images/{:03d}.png'.format(df['ImgId'][i]))))
		new_image.append(Image.open(os.path.join(args.output_directory , '{:03d}.png'.format(df['ImgId'][i]))))
		original_label.append(df['TrueLabel'][i])
	return (original_image , new_image , original_label)

def judge_1(new_image , original_label , black_box , device):
	transform = transforms.Compose([
		transforms.ToTensor() ,
		transforms.Normalize(mean = [0.485 , 0.456 , 0.406] , std = [0.229 , 0.224 , 0.225])
	])

	attack_dataset = dataset(new_image , original_label , transform)
	attack_dataloader = DataLoader(attack_dataset , batch_size = 1 , shuffle = False)

	black_box.to(device)
	black_box.eval()
	count = 0
	for (image , label) in attack_dataloader:
		(image , label) = (image.to(device) , label.to(device))
		y = black_box(image)
		(_ , index) = torch.max(y , dim = 1)
		if (index[0].item() != label.item()):
			count += 1
	
	print('success rate : {:.5f}'.format(count / len(new_image)))
	return

def judge_2(original_image , new_image):
	total_norm = 0
	for i in range(len(original_image)):
		total_norm += np.max(np.abs(np.asarray(original_image[i] , dtype = np.int) - np.asarray(new_image[i] , dtype = np.int)))
	print('average L-inf : {:.5f}'.format(total_norm / len(original_image)))
	return

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(original_image , new_image , original_label) = load_data(args)
	black_box = models.densenet121(pretrained = True)
	judge_1(new_image , original_label , black_box , device)
	judge_2(original_image , new_image)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)
