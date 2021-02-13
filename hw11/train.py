import os
import torch
from utils import my_argparse
from model import DCGAN , WGAN , LSGAN , SNGAN

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	if (args.model == 'DCGAN'):
		model = DCGAN(args.input_dim)
	if (args.model == 'WGAN'):
		model = WGAN(args.input_dim)
	if (args.model == 'WGAN-GP'):
		model = WGAN(args.input_dim , GP = True)
	if (args.model == 'LSGAN'):
		model = LSGAN(args.input_dim)
	if (args.model == 'SNGAN'):
		model = SNGAN(args.input_dim)
	model.train(args)
	torch.save(model.generator.state_dict() , args.checkpoint)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)