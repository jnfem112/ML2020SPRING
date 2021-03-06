import os
import numpy as np
import torch
from utils import my_argparse , get_dataloader
from data import load_dataset , dump
from model import CNN

def test(test_x , model , device):
	test_dataloader = get_dataloader(test_x , None , 'test')
	model.to(device)
	model.eval()
	test_y = list()
	with torch.no_grad():
		for data in test_dataloader:
			data = data.to(device)
			output = model(data)
			(_ , index) = torch.max(output , dim = 1)
			test_y.append(index.cpu().detach().numpy())
	test_y = np.concatenate(test_y , axis = 0)
	return test_y

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_x = load_dataset(None , None , args.test_directory)
	model = CNN()
	model.load_state_dict(torch.load('CNN.pkl' , map_location = device))
	test_y = test(test_x , model , device)
	dump(test_y , args.output_file)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)