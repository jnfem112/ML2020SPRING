import os
import numpy as np
import torch
from utils import my_argparse , get_dataloader
from data import load_data , dump
from model import RNN

def test(test_x , model , device):
	test_dataloader = get_dataloader(test_x , None , 'test')
	model.to(device)
	model.eval()
	test_y = list()
	with torch.no_grad():
		for data in test_dataloader:
			data = data.to(device , dtype = torch.long)
			output = model(data)
			output = output.squeeze()
			index = (output >= 0.5).int()
			test_y.append(index.cpu().detach().numpy())
	test_y = np.concatenate(test_y , axis = 0)
	return test_y

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(test_x , vocabulary) = load_data(None , None , args.test_data , args.max_length)
	model = RNN(vocabulary.embedding , vocabulary.token2index['<PAD>'])
	model.load_state_dict(torch.load('RNN.pkl' , map_location = device))
	test_y = test(test_x , model , device)
	dump(test_y , args.output_file)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)