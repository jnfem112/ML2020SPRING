import os
import numpy as np
import torch
from utils import my_argparse , get_dataloader
from data import load_data , dump
from model import StudentNet , decode8

def test(test_x , model , device):
	test_loader = get_dataloader(test_x , None , 'test')
	model.to(device)
	model.eval()
	test_y = list()
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)
			output = model(data)
			(_ , index) = torch.max(output , dim = 1)
			test_y.append(index.cpu().detach().numpy())
	test_y = np.concatenate(test_y , axis = 0)
	return test_y

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_x = load_data(None , None , args.test_directory)
	model = StudentNet()
	model.load_state_dict(decode8('StudentNet_encode8.pkl'))
	test_y = test(test_x , model , device)
	dump(test_y , args.output_file)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)