import os
import numpy as np
import torch
from torchvision import models
from utils import my_argparse , get_dataloader
from data import load_data

def judge_1(new_image , original_label , black_box , device):
	dataloader = get_dataloader(new_image , original_label)
	black_box.to(device)
	black_box.eval()
	count = 0
	for (image , label) in dataloader:
		(image , label) = (image.to(device) , label.to(device))
		output = black_box(image)
		(_ , index) = torch.max(output , dim = 1)
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
	(original_image , original_label , new_image) = load_data(args.input_directory , args.output_directory)
	black_box = models.densenet121(pretrained = True)
	judge_1(new_image , original_label , black_box , device)
	judge_2(original_image , new_image)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)