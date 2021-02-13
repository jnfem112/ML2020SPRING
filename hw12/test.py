import os
import numpy as np
import torch
from utils import my_argparse , get_dataloader
from data import dump
from model import Generator , Classifier

def test(generator , classifier_1 , classifier_2 , device , args):
	target_dataloader = get_dataloader(args.directory , 'test')
	generator.to(device)
	classifier_1.to(device)
	classifier_2.to(device)
	generator.eval()
	classifier_1.eval()
	classifier_2.eval()
	predict = list()
	with torch.no_grad():
		for (data , _) in target_dataloader:
			data = data.to(device)
			feature = generator(data)
			output_1 = classifier_1(feature)
			output_2 = classifier_2(feature)
			output = (output_1 + output_2) / 2
			(_ , index) = torch.max(output , dim = 1)
			predict.append(index.cpu().detach().numpy())
	predict = np.concatenate(predict , axis = 0)
	return predict

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(generator , classifier_1 , classifier_2) = (Generator() , Classifier() , Classifier())
	generator.load_state_dict(torch.load('generator.pkl' , map_location = device))
	classifier_1.load_state_dict(torch.load('classifier_1.pkl' , map_location = device))
	classifier_2.load_state_dict(torch.load('classifier_2.pkl' , map_location = device))
	predict = test(generator , classifier_1 , classifier_2 , device , args)
	dump(predict , args.output_file)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)