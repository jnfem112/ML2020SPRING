import numpy as np
import torch
import torch.nn as nn
import pickle

class StudentNet(nn.Module):
	def __init__(self , compression_ratio = 1):
		super(StudentNet , self).__init__()
		channel = [16 , 32 , 64 , 128 , 256 , 256 , 256 , 256]
		# 我們只prune第三層以後到最後一層以前的layer。
		for i in range(3 , len(channel) - 1):
			channel[i] = int(compression_ratio * channel[i])

		self.convolution = nn.Sequential(
			# 我們通常不會拆解第一層的convolutional layer。
			nn.Sequential(
				nn.Conv2d(3 , channel[0] , kernel_size = 3 , stride = 1 , padding = 1) ,
				nn.BatchNorm2d(channel[0]) ,
				nn.ReLU6() ,
				nn.MaxPool2d(2 , stride = 2 , padding = 0)
			) ,

			# 接下來每一個sequential block都一樣，所以我們只講解一個sequential block。
			nn.Sequential(
				# Depthwise convolution。
				nn.Conv2d(channel[0] , channel[0] , kernel_size = 3 , stride = 1 , padding = 1 , groups = channel[0]) ,
				# Batch normalization。
				nn.BatchNorm2d(channel[0]) ,
				# ReLU6是限制neuron的output最小只會到0，最大只會到6，MobileNet系列都是使用ReLU6。
 				# 使用ReLU6的原因是因為如果數字太大，qunatization時會不好壓到float16以下，因此才給個限制。
				nn.ReLU6() ,
				# Pointwise convolution。
				nn.Conv2d(channel[0] , channel[1] , kernel_size = 1) ,
				# 過完pointwise convolution不需要再經過ReLU，經驗上pointwise convolution加ReLU效果都會變差。
				nn.MaxPool2d(2 , stride = 2 , padding = 0)
				# 每過完一個block就down sampling。
			) ,

			nn.Sequential(
				nn.Conv2d(channel[1] , channel[1] , kernel_size = 3 , stride = 1 , padding = 1 , groups = channel[1]) ,
				nn.BatchNorm2d(channel[1]) ,
				nn.ReLU6() ,
				nn.Conv2d(channel[1] , channel[2] , kernel_size = 1) ,
				nn.MaxPool2d(2 , stride = 2 , padding = 0)
			) ,

			nn.Sequential(
				nn.Conv2d(channel[2] , channel[2] , kernel_size = 3 , stride = 1 , padding = 1 , groups = channel[2]) ,
				nn.BatchNorm2d(channel[2]) ,
				nn.ReLU6() ,
				nn.Conv2d(channel[2] , channel[3] , kernel_size = 1) ,
				nn.MaxPool2d(2 , stride = 2 , padding = 0)
			) ,

			# 到這邊為止因為圖片已經被down sample很多次了，所以就不再做MaxPool2d。
			nn.Sequential(
				nn.Conv2d(channel[3] , channel[3] , kernel_size = 3 , stride = 1 , padding = 1 , groups = channel[3]) ,
				nn.BatchNorm2d(channel[3]) ,
				nn.ReLU6() ,
				nn.Conv2d(channel[3] , channel[4] , kernel_size = 1)
			) ,

			nn.Sequential(
				nn.Conv2d(channel[4] , channel[4] , kernel_size = 3 , stride = 1 , padding = 1 , groups = channel[4]) ,
				nn.BatchNorm2d(channel[4]) ,
				nn.ReLU6() ,
				nn.Conv2d(channel[4] , channel[5] , kernel_size = 1)
			) ,

			nn.Sequential(
				nn.Conv2d(channel[5] , channel[5] , kernel_size = 3 , stride = 1 , padding = 1 , groups = channel[5]) ,
				nn.BatchNorm2d(channel[5]) ,
				nn.ReLU6() ,
				nn.Conv2d(channel[5] , channel[6] , kernel_size = 1)
			) ,

			nn.Sequential(
				nn.Conv2d(channel[6] , channel[6] , kernel_size = 3 , stride = 1 , padding = 1 , groups = channel[6]) ,
				nn.BatchNorm2d(channel[6]) ,
				nn.ReLU6() ,
				nn.Conv2d(channel[6] , channel[7] , kernel_size = 1)
			) ,

			# 這邊我們採用global average pooling。
			# 如果輸入圖片大小不一樣的話，就會因為global average pooling變成一樣的形狀。
			nn.AdaptiveAvgPool2d(1)
		)

		self.linear = nn.Sequential(
			nn.Linear(channel[7] , 11)
		)

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(x.size(0) , -1)
		x = self.linear(x)
		return x

def network_slimming(old_model , new_model):
	old_state_dict = old_model.state_dict()
	new_state_dict = new_model.state_dict()
	index = list()
	for i in range(len(old_model.convolution) - 1):
		importance = old_state_dict['convolution.{}.1.weight'.format(i)]
		rank = torch.argsort(importance , descending = True)
		dimension = len(new_state_dict['convolution.{}.1.weight'.format(i)])
		index.append(rank[ : dimension])

	layer = 1
	for (name , parameter) in old_state_dict.items():
		if (name.startswith('convolution') and parameter.dim() != 0 and layer < len(old_model.convolution) - 1):
			if (name.startswith('convolution.{}.3'.format(layer))):
				layer += 1
			if (name.endswith('3.weight')):
				if (layer == len(old_model.convolution) - 1):
					new_state_dict[name] = parameter[ : , index[layer - 1]]
				else:
					new_state_dict[name] = parameter[index[layer]][ : ,index[layer - 1]]
			else:
				new_state_dict[name] = parameter[index[layer]]
		else:
			new_state_dict[name] = parameter

	new_model.load_state_dict(new_state_dict)
	return new_model

def encode16(state_dict , path):
	compress_state_dict = dict()
	for (name , parameter) in state_dict.items():
		parameter = np.float64(parameter.cpu().numpy())
		if (type(parameter) == np.ndarray):
			compress_state_dict[name] = np.float16(parameter)
		else:
			compress_state_dict[name] = parameter

	with open(path , 'wb') as file:
		pickle.dump(compress_state_dict , file)

	return

def decode16(path):
	with open(path , 'rb') as file:
		compress_state_dict = pickle.load(file)

	state_dict = dict()
	for (name , parameter) in compress_state_dict.items():
		parameter = torch.tensor(parameter)
		state_dict[name] = parameter

	return state_dict

def encode8(state_dict , path):
	compress_state_dict = dict()
	for (name , parameter) in state_dict.items():
		parameter = np.float64(parameter.cpu().numpy())
		if (type(parameter) == np.ndarray):
			min_value = np.min(parameter)
			max_value = np.max(parameter)
			parameter = np.round(255 * (parameter - min_value) / (max_value - min_value))
			parameter = np.uint8(parameter)
			compress_state_dict[name] = (min_value , max_value , parameter)
		else:
			compress_state_dict[name] = parameter

	with open(path , 'wb') as file:
		pickle.dump(compress_state_dict , file)

	return

def decode8(path):
	with open(path , 'rb') as file:
		compress_state_dict = pickle.load(file)

	state_dict = dict()
	for (name , parameter) in compress_state_dict.items():
		if (type(parameter) == tuple):
			(min_value , max_value , parameter) = parameter
			parameter = np.float64(parameter)
			parameter = parameter / 255 * (max_value - min_value) + min_value
			parameter = torch.tensor(parameter)
		else:
			parameter = torch.tensor(parameter)
		state_dict[name] = parameter

	return state_dict