import os
import torch
from torch.autograd import Variable , grad
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.optim import RMSprop , Adam
from torchvision.utils import save_image
from time import time
from utils import set_random_seed , get_dataloader , print_progress

def init_weight(layer):
	if (layer.__class__.__name__.find('Conv') != -1):
		layer.weight.data.normal_(0 , 0.02)
	elif (layer.__class__.__name__.find('BatchNorm') != -1):
		layer.weight.data.normal_(1 , 0.02)
		layer.bias.data.fill_(0)
	return

################################################## DCGAN ##################################################

class DCGAN_generator(nn.Module):
	def __init__(self , input_dim):
		super(DCGAN_generator , self).__init__()

		self.linear = nn.Sequential(
				nn.Linear(input_dim , 8192 , bias = False) ,
				nn.BatchNorm1d(8192) ,
				nn.ReLU()
			)

		self.deconvolution = nn.Sequential(
			nn.ConvTranspose2d(512 , 256 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(256 , 128 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(128 , 64 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(64 , 3 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1) ,
			nn.Tanh()
		)

		self.apply(init_weight)
		return

	def forward(self , x):
		x = self.linear(x)
		x = x.view(x.size(dim = 0) , -1 , 4 , 4)
		x = self.deconvolution(x)
		return x

class DCGAN_discriminator(nn.Module):
	def __init__(self):
		super(DCGAN_discriminator , self).__init__()

		self.convolution = nn.Sequential(
				nn.Conv2d(3 , 64 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(64 , 128 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.BatchNorm2d(128) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(128 , 256 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.BatchNorm2d(256) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(256 , 512 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.BatchNorm2d(512) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(512 , 1 , kernel_size = 4) ,
				nn.Sigmoid()
			)

		self.apply(init_weight)
		return

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(-1)
		return x

class DCGAN():
	def __init__(self , input_dim):
		self.generator = DCGAN_generator(input_dim)
		self.discriminator = DCGAN_discriminator()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		return

	def train(self , args):
		# Hyper-parameter.
		train_directory = args.train_directory
		number_of_data = len(os.listdir(train_directory))
		input_dim = args.input_dim
		batch_size = args.batch_size
		learning_rate = args.learning_rate
		epoch = args.epoch

		set_random_seed(0)

		train_dataloader = get_dataloader(train_directory , batch_size)
		self.generator = self.generator.to(self.device)
		self.discriminator = self.discriminator.to(self.device)
		optimizer_generator = Adam(self.generator.parameters() , lr = learning_rate , betas = (0.5 , 0.999))
		optimizer_discriminator = Adam(self.discriminator.parameters() , lr = learning_rate , betas = (0.5 , 0.999))
		criterion = nn.BCELoss()
		test_input = Variable(torch.randn(100 , input_dim)).to(self.device)
		for i in range(epoch):
			self.generator.train()
			self.discriminator.train()
			total_loss_generator = 0
			total_loss_discriminator = 0
			start = time()
			for (j , image) in enumerate(train_dataloader):
				# Update discriminator.
				input = Variable(torch.randn(image.size(dim = 0) , input_dim)).to(self.device)
				real_image = Variable(image).to(self.device)
				fake_image = self.generator(input)
				real_label = torch.ones((image.size(dim = 0))).to(self.device)
				fake_label = torch.zeros((image.size(dim = 0))).to(self.device)
				optimizer_discriminator.zero_grad()
				real_output = self.discriminator(real_image.detach())
				fake_output = self.discriminator(fake_image.detach())
				loss_discriminator = (criterion(real_output , real_label) + criterion(fake_output , fake_label)) / 2
				total_loss_discriminator += loss_discriminator.item()
				loss_discriminator.backward()
				optimizer_discriminator.step()
				# Update generator.
				input = Variable(torch.randn(image.size(dim = 0) , input_dim)).to(self.device)
				fake_image = self.generator(input)
				fake_output = self.discriminator(fake_image)
				optimizer_generator.zero_grad()
				loss_generator = criterion(fake_output , real_label)
				total_loss_generator += loss_generator.item()
				loss_generator.backward()
				optimizer_generator.step()
				end = time()
				print_progress(i + 1 , epoch , number_of_data , batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss_generator / number_of_data , total_loss_discriminator / number_of_data)

			fake_image = self.inference(test_input)
			save_image(fake_image , 'DCGAN_epoch_{}.jpg'.format(i + 1) , nrow = 10)

		return

	def inference(self , input):
		input = input.to(self.device)
		self.generator.to(self.device)
		self.generator.eval()
		start = time()
		output = self.generator(input).data
		image = (output + 1) / 2
		end = time()
		print('inference ({}s)'.format(int(end - start)))
		return image

	def save(self):
		torch.save(self.generator.state_dict() , 'DCGAN_generator.pkl')
		torch.save(self.discriminator.state_dict() , 'DCGAN_discriminator.pkl')
		return

	def load(self):
		self.generator.load_state_dict(torch.load('DCGAN_generator.pkl' , map_location = self.device))
		self.discriminator.load_state_dict(torch.load('DCGAN_discriminator.pkl' , map_location = self.device))
		return

################################################## WGAN & WGAN-GP ##################################################

# https://github.com/arturml/pytorch-wgan-gp
class WGAN_generator(nn.Module):
	def __init__(self , input_dim):
		super(WGAN_generator , self).__init__()

		self.linear = nn.Sequential(
				nn.Linear(input_dim , 8192 , bias = False) ,
				nn.BatchNorm1d(8192) ,
				nn.ReLU()
			)

		self.deconvolution = nn.Sequential(
			nn.ConvTranspose2d(512 , 256 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(256 , 128 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(128 , 64 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(64 , 3 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1) ,
			nn.Tanh()
		)

		self.apply(init_weight)
		return

	def forward(self , x):
		x = self.linear(x)
		x = x.view(x.size(dim = 0) , -1 , 4 , 4)
		x = self.deconvolution(x)
		return x

class WGAN_discriminator(nn.Module):
	def __init__(self):
		super(WGAN_discriminator , self).__init__()

		self.convolution = nn.Sequential(
				nn.Conv2d(3 , 64 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(64 , 128 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.InstanceNorm2d(128) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(128 , 256 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.InstanceNorm2d(256) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(256 , 512 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.InstanceNorm2d(512) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(512 , 1 , kernel_size = 4)
			)

		self.apply(init_weight)
		return

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(-1)
		return x

class WGAN():
	def __init__(self , input_dim , GP = False):
		self.generator = WGAN_generator(input_dim)
		self.discriminator = WGAN_discriminator()
		self.GP = GP
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		return

	def gradient_penalty(self , real_image , fake_image):
		weight = torch.rand((real_image.size(dim = 0) , 1 , 1 , 1) , device = self.device)
		input = Variable(weight * real_image + (1 - weight) * fake_image , requires_grad = True)
		output = self.discriminator(input)
		grad_output = Variable(torch.ones(real_image.size(dim = 0)) , requires_grad = False).to(self.device)
		gradient = grad(outputs = output , inputs = input , grad_outputs = grad_output , create_graph = True , retain_graph = True , only_inputs = True)[0]
		gradient = gradient.view(gradient.size(dim = 0) , -1)
		return torch.mean((gradient.norm(2 , dim = 1) - 1)**2)

	def train(self , args):
		# Hyper-parameter.
		train_directory = args.train_directory
		number_of_data = len(os.listdir(train_directory))
		input_dim = args.input_dim
		batch_size = args.batch_size
		learning_rate = args.learning_rate
		clip_value = args.clip_value
		lambd = args.lambd
		n_critic = args.n_critic
		epoch = args.epoch

		set_random_seed(0)

		train_dataloader = get_dataloader(train_directory , batch_size)
		self.generator = self.generator.to(self.device)
		self.discriminator = self.discriminator.to(self.device)
		if (not self.GP):
			optimizer_generator = RMSprop(self.generator.parameters() , lr = learning_rate)
			optimizer_discriminator = RMSprop(self.discriminator.parameters() , lr = learning_rate)
		else:
			optimizer_generator = Adam(self.generator.parameters() , lr = learning_rate , betas = (0.5 , 0.999))
			optimizer_discriminator = Adam(self.discriminator.parameters() , lr = learning_rate , betas = (0.5 , 0.999))
		test_input = Variable(torch.randn(100 , input_dim)).to(self.device)
		for i in range(epoch):
			self.generator.train()
			self.discriminator.train()
			total_loss_generator = 0
			total_loss_discriminator = 0
			start = time()
			for (j , image) in enumerate(train_dataloader):
				# Update discriminator.
				input = Variable(torch.randn(image.size(dim = 0) , input_dim)).to(self.device)
				real_image = Variable(image).to(self.device)
				fake_image = self.generator(input)
				optimizer_discriminator.zero_grad()
				real_output = self.discriminator(real_image.detach())
				fake_output = self.discriminator(fake_image.detach())
				if (not self.GP):
					loss_discriminator = -torch.mean(real_output) + torch.mean(fake_output)
				else:
					loss_discriminator = -torch.mean(real_output) + torch.mean(fake_output) + lambd * gradient_penalty(real_image , fake_image)
				total_loss_discriminator += loss_discriminator.item()
				loss_discriminator.backward()
				optimizer_discriminator.step()
				if (not self.GP):
					for weight in self.discriminator.parameters():
						weight.data.clamp_(-clip_value , clip_value)
				# Update generator.
				if ((j + 1) % n_critic == 0):
					input = Variable(torch.randn(image.size(dim = 0) , input_dim)).to(self.device)
					fake_image = self.generator(input)
					fake_output = self.discriminator(fake_image)
					optimizer_generator.zero_grad()
					loss_generator = -torch.mean(fake_output)
					total_loss_generator += loss_generator.item()
					loss_generator.backward()
					optimizer_generator.step()
				end = time()
				print_progress(i + 1 , epoch , number_of_data , batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss_generator , total_loss_discriminator)

			fake_image = self.inference(test_input)
			if (not self.GP):
				save_image(fake_image , 'WGAN_epoch_{}.jpg'.format(i + 1) , nrow = 10)
			else:
				save_image(fake_image , 'WGAN-GP_epoch_{}.jpg'.format(i + 1) , nrow = 10)

		return

	def inference(self , input):
		input = input.to(self.device)
		self.generator.to(self.device)
		self.generator.eval()
		start = time()
		output = self.generator(input).data
		image = (output + 1) / 2
		end = time()
		print('inference ({}s)'.format(int(end - start)))
		return image

	def save(self):
		if (not self.GP):
			torch.save(self.generator.state_dict() , 'WGAN_generator.pkl')
			torch.save(self.discriminator.state_dict() , 'WGAN_discriminator.pkl')
		else:
			torch.save(self.generator.state_dict() , 'WGAN-GP_generator.pkl')
			torch.save(self.discriminator.state_dict() , 'WGAN-GP_discriminator.pkl')
		return

	def load(self):
		if (not self.GP):
			self.generator.load_state_dict(torch.load('WGAN_generator.pkl' , map_location = self.device))
			self.discriminator.load_state_dict(torch.load('WGAN_discriminator.pkl' , map_location = self.device))
		else:
			self.generator.load_state_dict(torch.load('WGAN-GP_generator.pkl' , map_location = self.device))
			self.discriminator.load_state_dict(torch.load('WGAN-GP_discriminator.pkl' , map_location = self.device))
		return

################################################## LSGAN ##################################################

class LSGAN_generator(nn.Module):
	def __init__(self , input_dim):
		super(LSGAN_generator , self).__init__()

		self.linear = nn.Sequential(
				nn.Linear(input_dim , 8192 , bias = False) ,
				nn.BatchNorm1d(8192) ,
				nn.ReLU()
			)

		self.deconvolution = nn.Sequential(
			nn.ConvTranspose2d(512 , 256 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(256 , 128 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(128 , 64 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(64 , 3 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1) ,
			nn.Tanh()
		)

		self.apply(init_weight)
		return

	def forward(self , x):
		x = self.linear(x)
		x = x.view(x.size(dim = 0) , -1 , 4 , 4)
		x = self.deconvolution(x)
		return x

class LSGAN_discriminator(nn.Module):
	def __init__(self):
		super(LSGAN_discriminator , self).__init__()

		self.convolution = nn.Sequential(
				nn.Conv2d(3 , 64 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(64 , 128 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.BatchNorm2d(128) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(128 , 256 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.BatchNorm2d(256) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(256 , 512 , kernel_size = 5 , stride = 2 , padding = 2) ,
				nn.BatchNorm2d(512) ,
				nn.LeakyReLU(0.2) ,
				nn.Conv2d(512 , 1 , kernel_size = 4)
			)

		self.apply(init_weight)
		return

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(-1)
		return x

class LSGAN():
	def __init__(self , input_dim):
		self.generator = LSGAN_generator(input_dim)
		self.discriminator = LSGAN_discriminator()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		return

	def train(self , args):
		# Hyper-parameter.
		train_directory = args.train_directory
		number_of_data = len(os.listdir(train_directory))
		input_dim = args.input_dim
		batch_size = args.batch_size
		learning_rate = args.learning_rate
		epoch = args.epoch

		set_random_seed(0)

		train_dataloader = get_dataloader(train_directory , batch_size)
		self.generator = self.generator.to(self.device)
		self.discriminator = self.discriminator.to(self.device)
		optimizer_generator = Adam(self.generator.parameters() , lr = learning_rate , betas = (0.5 , 0.999))
		optimizer_discriminator = Adam(self.discriminator.parameters() , lr = learning_rate , betas = (0.5 , 0.999))
		criterion = nn.MSELoss()
		test_input = Variable(torch.randn(100 , input_dim)).to(self.device)
		for i in range(epoch):
			self.generator.train()
			self.discriminator.train()
			total_loss_generator = 0
			total_loss_discriminator = 0
			start = time()
			for (j , image) in enumerate(train_dataloader):
				# Update discriminator.
				input = Variable(torch.randn(image.size(dim = 0) , input_dim)).to(self.device)
				real_image = Variable(image).to(self.device)
				fake_image = self.generator(input)
				real_label = torch.ones((image.size(dim = 0))).to(self.device)
				fake_label = torch.zeros((image.size(dim = 0))).to(self.device)
				optimizer_discriminator.zero_grad()
				real_output = self.discriminator(real_image.detach())
				fake_output = self.discriminator(fake_image.detach())
				loss_discriminator = (criterion(real_output , real_label) + criterion(fake_output , fake_label)) / 2
				total_loss_discriminator += loss_discriminator.item()
				loss_discriminator.backward()
				optimizer_discriminator.step()
				# Update generator.
				input = Variable(torch.randn(image.size(dim = 0) , input_dim)).to(self.device)
				fake_image = self.generator(input)
				fake_output = self.discriminator(fake_image)
				optimizer_generator.zero_grad()
				loss_generator = criterion(fake_output , real_label)
				total_loss_generator += loss_generator.item()
				loss_generator.backward()
				optimizer_generator.step()
				end = time()
				print_progress(i + 1 , epoch , number_of_data , batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss_generator / number_of_data , total_loss_discriminator / number_of_data)

			fake_image = self.inference(test_input)
			save_image(fake_image , 'LSGAN_epoch_{}.jpg'.format(i + 1) , nrow = 10)

		return

	def inference(self , input):
		input = input.to(self.device)
		self.generator.to(self.device)
		self.generator.eval()
		start = time()
		output = self.generator(input).data
		image = (output + 1) / 2
		end = time()
		print('inference ({}s)'.format(int(end - start)))
		return image

	def save(self):
		torch.save(self.generator.state_dict() , 'LSGAN_generator.pkl')
		torch.save(self.discriminator.state_dict() , 'LSGAN_discriminator.pkl')
		return

	def load(self):
		self.generator.load_state_dict(torch.load('LSGAN_generator.pkl' , map_location = self.device))
		self.discriminator.load_state_dict(torch.load('LSGAN_discriminator.pkl' , map_location = self.device))
		return

################################################## SNGAN ##################################################

class SNGAN_generator(nn.Module):
	def __init__(self , input_dim):
		super(SNGAN_generator , self).__init__()

		self.linear = nn.Sequential(
				nn.Linear(input_dim , 8192 , bias = False) ,
				nn.BatchNorm1d(8192) ,
				nn.ReLU()
			)

		self.deconvolution = nn.Sequential(
			nn.ConvTranspose2d(512 , 256 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(256 , 128 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(128 , 64 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1 , bias = False) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(64 , 3 , kernel_size = 5 , stride = 2 , padding = 2 , output_padding = 1) ,
			nn.Tanh()
		)

		self.apply(init_weight)
		return

	def forward(self , x):
		x = self.linear(x)
		x = x.view(x.size(dim = 0) , -1 , 4 , 4)
		x = self.deconvolution(x)
		return x

class SNGAN_discriminator(nn.Module):
	def __init__(self):
		super(SNGAN_discriminator , self).__init__()

		self.convolution = nn.Sequential(
				spectral_norm(nn.Conv2d(3 , 64 , kernel_size = 5 , stride = 2 , padding = 2)) ,
				nn.LeakyReLU(0.2) ,
				spectral_norm(nn.Conv2d(64 , 128 , kernel_size = 5 , stride = 2 , padding = 2)) ,
				nn.LeakyReLU(0.2) ,
				spectral_norm(nn.Conv2d(128 , 256 , kernel_size = 5 , stride = 2 , padding = 2)) ,
				nn.LeakyReLU(0.2) ,
				spectral_norm(nn.Conv2d(256 , 512 , kernel_size = 5 , stride = 2 , padding = 2)) ,
				nn.LeakyReLU(0.2) ,
				spectral_norm(nn.Conv2d(512 , 1 , kernel_size = 4)) ,
				nn.Sigmoid()
			)

		self.apply(init_weight)
		return

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(-1)
		return x

class SNGAN():
	def __init__(self , input_dim):
		self.generator = SNGAN_generator(input_dim)
		self.discriminator = SNGAN_discriminator()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		return

	def train(self , args):
		# Hyper-parameter.
		train_directory = args.train_directory
		number_of_data = len(os.listdir(train_directory))
		input_dim = args.input_dim
		batch_size = args.batch_size
		learning_rate = args.learning_rate
		epoch = args.epoch

		set_random_seed(0)

		train_dataloader = get_dataloader(train_directory , batch_size)
		self.generator = self.generator.to(self.device)
		self.discriminator = self.discriminator.to(self.device)
		optimizer_generator = Adam(self.generator.parameters() , lr = learning_rate , betas = (0.5 , 0.999))
		optimizer_discriminator = Adam(self.discriminator.parameters() , lr = learning_rate , betas = (0.5 , 0.999))
		criterion = nn.BCELoss()
		test_input = Variable(torch.randn(100 , input_dim)).to(self.device)
		for i in range(epoch):
			self.generator.train()
			self.discriminator.train()
			total_loss_generator = 0
			total_loss_discriminator = 0
			start = time()
			for (j , image) in enumerate(train_dataloader):
				# Update discriminator.
				input = Variable(torch.randn(image.size(dim = 0) , input_dim)).to(self.device)
				real_image = Variable(image).to(self.device)
				fake_image = self.generator(input)
				real_label = torch.ones((image.size(dim = 0))).to(self.device)
				fake_label = torch.zeros((image.size(dim = 0))).to(self.device)
				optimizer_discriminator.zero_grad()
				real_output = self.discriminator(real_image.detach())
				fake_output = self.discriminator(fake_image.detach())
				loss_discriminator = (criterion(real_output , real_label) + criterion(fake_output , fake_label)) / 2
				total_loss_discriminator += loss_discriminator.item()
				loss_discriminator.backward()
				optimizer_discriminator.step()
				# Update generator.
				input = Variable(torch.randn(image.size(dim = 0) , input_dim)).to(self.device)
				fake_image = self.generator(input)
				fake_output = self.discriminator(fake_image)
				optimizer_generator.zero_grad()
				loss_generator = criterion(fake_output , real_label)
				total_loss_generator += loss_generator.item()
				loss_generator.backward()
				optimizer_generator.step()
				end = time()
				print_progress(i + 1 , epoch , number_of_data , batch_size , j + 1 , len(train_dataloader) , int(end - start) , total_loss_generator / number_of_data , total_loss_discriminator / number_of_data)

			fake_image = self.inference(test_input)
			save_image(fake_image , 'SNGAN_epoch_{}.jpg'.format(i + 1) , nrow = 10)

		return

	def inference(self , input):
		input = input.to(self.device)
		self.generator.to(self.device)
		self.generator.eval()
		start = time()
		output = self.generator(input).data
		image = (output + 1) / 2
		end = time()
		print('inference ({}s)'.format(int(end - start)))
		return image

	def save(self):
		torch.save(self.generator.state_dict() , 'SNGAN_generator.pkl')
		torch.save(self.discriminator.state_dict() , 'SNGAN_discriminator.pkl')
		return

	def load(self):
		self.generator.load_state_dict(torch.load('SNGAN_generator.pkl' , map_location = self.device))
		self.discriminator.load_state_dict(torch.load('SNGAN_discriminator.pkl' , map_location = self.device))
		return