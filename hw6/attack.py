import os
import torch
from torchvision import models
import torch.nn as nn
from time import time
from utils import my_argparse , get_dataloader , deprocess
from data import load_data , save_image

def FGSM(original_image , original_label , proxy_model , device , args):
	# Hyper-parameter.
	epsilon = args.epsilon

	dataloader = get_dataloader(original_image , original_label)
	proxy_model.to(device)
	proxy_model.eval()
	criterion = nn.CrossEntropyLoss()
	new_image = list()
	new_label = list()
	for (i , (image , label)) in enumerate(dataloader):
		print('image {}'.format(i + 1) , end = ' ')
		start = time()
		(image , label) = (image.to(device) , label.to(device))
		image.requires_grad = True
		output = proxy_model(image)
		(_ , index) = torch.max(output , dim = 1)
		loss = criterion(output , label)
		loss.backward()
		if (index[0].item() == label.item()):
			image = image + epsilon * torch.sign(image.grad)
		end = time()
		print('({}s)'.format(int(end - start)))

		output = proxy_model(image)
		(_ , index) = torch.max(output , dim = 1)
		new_label.append(index[0].item())

		image = image.cpu().detach().squeeze(dim = 0).numpy()
		image = deprocess(image)
		new_image.append(image)

	return (new_image , new_label)

def BIM(original_image , original_label , proxy_model , device , args):
	# Hyper-parameter.
	epsilon = args.epsilon

	dataloader = get_dataloader(original_image , original_label)
	proxy_model.to(device)
	proxy_model.eval()
	criterion = nn.CrossEntropyLoss()
	new_image = list()
	new_label = list()
	for (i , (image , label)) in enumerate(dataloader):
		print('image {}'.format(i + 1) , end = ' ')
		start = time()
		(image , label) = (image.to(device) , label.to(device))
		while True:
			image.requires_grad = True
			output = proxy_model(image)
			(_ , index) = torch.max(output , dim = 1)
			loss = criterion(output , label)
			loss.backward()
			if (index[0].item() != label.item()):
				break
			else:
				image = nn.Parameter(image + epsilon * torch.sign(image.grad))
		end = time()
		print('({}s)'.format(int(end - start)))

		image = image.cpu().detach().squeeze(dim = 0).numpy()
		image = deprocess(image)
		new_image.append(image)
		new_label.append(index[0].item())

	return (new_image , new_label)

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(original_image , original_label) = load_data(args.input_directory)
	proxy_model = models.densenet121(pretrained = True)
	if (args.method == 'FGSM'):
		(new_image , new_label) = FGSM(original_image , original_label , proxy_model , device , args)
	else:
		(new_image , new_label) = BIM(original_image , original_label , proxy_model , device , args)
	save_image(new_image , args.output_directory)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)