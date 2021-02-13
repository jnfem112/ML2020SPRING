import argparse
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset , DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD , Adam
from time import time
from copy import deepcopy
import matplotlib.pyplot as plt

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--number_of_task' , type = int , default = 500000)
	parser.add_argument('--batch_size' , type = int , default = 10)
	parser.add_argument('--learning_rate' , type = float , default = 0.001)
	parser.add_argument('--epoch' , type = int , default = 5)
	args = parser.parse_args()
	return args

def generate_dataset(seed = 0 , min_a = 0.1 , max_a = 5 , min_b =  0 , max_b = 2 * np.pi , number_of_task = 100 , min_x = -5 , max_x = 5 , number_of_sample = 10 , plot = False):
	np.random.seed(seed)

	a = np.random.uniform(min_a , max_a , number_of_task)
	b = np.random.uniform(min_b , max_b , number_of_task)

	x = list()
	y = list()
	for i in range(number_of_task):
		x.append(np.random.uniform(min_x , max_x , number_of_sample))
		y.append(a[i] * np.sin(x[-1] + b[i]))

	if (plot):
		plot_x = np.linspace(min_x , max_x , 1000)
		plot_y = list()
		for i in range(number_of_task):
			plot_y.append(a[i] * np.sin(plot_x + b[i]))

	return (x , y) if (not plot) else (x , y , plot_x , plot_y)

def get_dataloader(x , y , mode , batch_size = 1024 , num_workers = 8):
	x = torch.Tensor(x).unsqueeze(dim = -1)
	y = torch.Tensor(y).unsqueeze(dim = -1)
	dataset = TensorDataset(x , y)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

class MetaLinear(nn.Module):
	def __init__(self , init_layer = None):
		super(MetaLinear , self).__init__()
		if (type(init_layer) != type(None)):
			self.weight = init_layer.weight.clone()
			self.bias = init_layer.bias.clone()
		return

	def zero_grad(self):
		self.weight.grad = torch.zeros_like(self.weight)
		self.bias.grad = torch.zeros_like(self.bias)
		return

	def forward(self , x):
		return F.linear(x , self.weight , self.bias)

class Net(nn.Module):
	def __init__(self , init_weight = None):
		super(Net , self).__init__()
		if (type(init_weight) != type(None)):
			for (name , module) in init_weight.named_modules():
				if (name):
					setattr(self , name , MetaLinear(module))
		else:
			self.hidden1 = nn.Linear(1 , 40)
			self.hidden2 = nn.Linear(40 , 40)
			self.out = nn.Linear(40 , 1)
		return
	
	def zero_grad(self):
		layers = self.__dict__['_modules']
		for layer in layers.keys():
			layers[layer].zero_grad()
		return
		
	def forward(self , x):
		x = F.relu(self.hidden1(x))
		x = F.relu(self.hidden2(x))
		return self.out(x)

	def update(self , parent , lr = 1):
		layers = self.__dict__['_modules']
		parent_layers = parent.__dict__['_modules']
		for param in layers.keys():
			layers[param].weight = layers[param].weight - lr * parent_layers[param].weight.grad
			layers[param].bias = layers[param].bias - lr * parent_layers[param].bias.grad
		return

class MetaNet():
	def __init__(self , init_weight = None):
		super(MetaNet , self).__init__()
		self.model = Net()
		if (type(init_weight) != type(None)):
			self.model.load_state_dict(init_weight)
		return

	def gen_models(self , number_of_model , check = True):
		models = [Net(init_weight = self.model) for i in range(number_of_model)]
		return models

def train(train_x , train_y , pretrained_model , meta_model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate

	train_dataloader = get_dataloader(train_x , train_y , 'train' , batch_size)
	pretrained_model.to(device)
	meta_model.model.to(device)
	pretrained_model.train()
	meta_model.model.train()
	pretrained_optimizer = Adam(pretrained_model.parameters() , lr = learning_rate)
	meta_optimizer = Adam(meta_model.model.parameters() , lr = learning_rate)
	criterion = nn.MSELoss()
	for i in range(1):
		start = time()
		for (j , (x , y)) in enumerate(train_dataloader):
			(x , y) = (x.to(device) , y.to(device))
			sub_model = meta_model.gen_models(batch_size)
			total_meta_loss = 0
			for k in range(batch_size):
				index = np.arange(batch_size)
				np.random.shuffle(index)
				# Update pretrained model.
				pretrained_optimizer.zero_grad()
				pretrained_output = pretrained_model(x[k][index[ : batch_size // 2]])
				pretrained_loss = criterion(pretrained_output , y[k][index[ : batch_size // 2]])
				pretrained_loss.backward()
				pretrained_optimizer.step()
				pretrained_optimizer.zero_grad()
				pretrained_output = pretrained_model(x[k][index[batch_size // 2 : ]])
				pretrained_loss = criterion(pretrained_output , y[k][index[batch_size // 2 : ]])
				pretrained_loss.backward()
				pretrained_optimizer.step()
				# Update Meta model.
				sub_model[k].to(device)
				meta_output = sub_model[k](x[k][index[ : batch_size // 2]])
				meta_loss = criterion(meta_output , y[k][index[ : batch_size // 2]])
				meta_loss.backward(create_graph = False)
				sub_model[k].update(lr = learning_rate , parent = meta_model.model)
				meta_optimizer.zero_grad()
				meta_output = sub_model[k](x[k][index[batch_size // 2 : ]])
				meta_loss = criterion(meta_output , y[k][index[batch_size // 2 : ]])
				total_meta_loss += meta_loss
			meta_loss = total_meta_loss / batch_size
			meta_loss.backward()
			meta_optimizer.step()
			meta_optimizer.zero_grad()
			print('\rbatch {}/{}'.format(j + 1 , len(train_dataloader)) , end = '')
		end = time()
		print(' ({}s)'.format(int(end - start)))

	return (pretrained_model , meta_model)

def test(test_x , test_y , pretrained_model , meta_model , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	epoch = args.epoch

	test_dataloader = get_dataloader(test_x , test_y , 'test' , batch_size)
	model_1 = deepcopy(pretrained_model)
	model_1.to(device)
	model_1.train()
	optimizer = SGD(model_1.parameters() , lr = learning_rate)
	criterion = nn.MSELoss()
	print('(pretrained model)')
	for i in range(epoch):
		start = time()
		for (x , y) in test_dataloader:
			(x , y) = (x.to(device) , y.to(device))
			optimizer.zero_grad()
			output = model_1(x[0])
			loss = criterion(output , y[0])
			loss.backward()
			optimizer.step()
		end = time()
		print('epoch {}/{} ({}s) loss : {:.8f}'.format(i + 1 , epoch , int(end - start) , loss.item()))

	test_dataloader = get_dataloader(test_x , test_y , 'test' , batch_size)
	model_2 = deepcopy(meta_model.model)
	model_2.to(device)
	model_2.train()
	optimizer = SGD(model_2.parameters() , lr = learning_rate)
	criterion = nn.MSELoss()
	print('\n(meta model)')
	for i in range(epoch):
		start = time()
		for (x , y) in test_dataloader:
			(x , y) = (x.to(device) , y.to(device))
			optimizer.zero_grad()
			output = model_2(x[0])
			loss = criterion(output , y[0])
			loss.backward()
			optimizer.step()
		end = time()
		print('epoch {}/{} ({}s) loss : {:.8f}'.format(i + 1 , epoch , int(end - start) , loss.item()))

	return (model_1 , model_2)

def plot(test_x , test_y , plot_x , plot_y , model_1 , model_2 , device):
	plt.figure(figsize = (12 , 9))
	plt.scatter(test_x , test_y)
	plt.plot(plot_x , plot_y)

	x = torch.Tensor(plot_x).unsqueeze(dim = -1)
	x = x.to(device)

	model_1.to(device)
	model_1.eval()
	y = model_1(x)
	plot_y = y.cpu().detach().squeeze().numpy()
	plt.plot(plot_x , plot_y , label = 'pretrained model' , c = 'red')

	model_2.to(device)
	model_2.eval()
	y = model_2(x)
	plot_y = y.cpu().detach().squeeze().numpy()
	plt.plot(plot_x , plot_y , label = 'meta model' , c = 'orange')

	plt.legend(loc = 'lower right' , fontsize = 16)
	plt.show()
	return

def main(args):
	print('============ setup device ============')
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('========== generate dataset ==========')
	(train_x , train_y) = generate_dataset(number_of_task = args.number_of_task)
	(test_x , test_y , plot_x , plot_y) = generate_dataset(number_of_task = 1 , plot = True)
	print('============ create model ============')
	(pretrained_model , meta_model) = (Net() , MetaNet())
	print('=============== train ================')
	(pretrained_model , meta_model) = train(train_x , train_y , pretrained_model , meta_model , device , args)
	print('================ test ================')
	(model_1 , model_2) = test(test_x , test_y , pretrained_model , meta_model , device , args)
	plot(test_x , test_y , plot_x , plot_y[0] , model_1 , model_2 , device)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)