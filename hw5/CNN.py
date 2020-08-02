import torch.nn as nn

class CNN(nn.Module):
	def __init__(self):
		super(CNN , self).__init__()

		self.convolution = nn.Sequential(
			nn.Conv2d(3 , 64 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1)) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.Conv2d(64 , 64 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1)) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d(p = 0.1) ,
			nn.Conv2d(64 , 128 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1)) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.Conv2d(128 , 128 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1)) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d(p = 0.1) ,
			nn.Conv2d(128 , 256 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1)) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.Conv2d(256 , 256 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1)) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d(p = 0.1) ,
			nn.Conv2d(256 , 512 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1)) ,
			nn.BatchNorm2d(512) ,
			nn.ReLU() ,
			nn.Conv2d(512 , 512 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1)) ,
			nn.BatchNorm2d(512) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d(p = 0.5)
		)

		self.linear = nn.Sequential(
			nn.Linear(32768 , 1024 , bias = True) ,
			nn.BatchNorm1d(1024) ,
			nn.ReLU() ,
			nn.Dropout(p = 0.5) ,
			nn.Linear(1024 , 11 , bias = True)
		)

		return

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(x.size(0) , -1)
		x = self.linear(x)
		return x