import os
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
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
	model.generator.load_state_dict(torch.load(args.checkpoint))
	input = Variable(torch.randn(100 , input_dim))
	image = model.inference(input)
	save_image(image , args.output_image , nrow = 10)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)