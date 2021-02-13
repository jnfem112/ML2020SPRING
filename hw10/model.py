import torch.nn as nn

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder , self).__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(3 , 16 , kernel_size = 4 , stride = 2 , padding = 1) ,
			nn.BatchNorm2d(16) ,
			nn.ReLU() ,
			nn.Conv2d(16 , 32 , kernel_size = 4 , stride = 2 , padding = 1) ,
			nn.BatchNorm2d(32) ,
			nn.ReLU() ,
			nn.Conv2d(32 , 64 , kernel_size = 4 , stride = 2 , padding = 1) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU()
		)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(64 , 32 , kernel_size = 4 , stride = 2 , padding = 1) ,
			nn.BatchNorm2d(32) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(32 , 16 , kernel_size = 4 , stride = 2 , padding = 1) ,
			nn.BatchNorm2d(16) ,
			nn.ReLU() ,
			nn.ConvTranspose2d(16 , 3 , kernel_size = 4 , stride = 2 , padding = 1) ,
			nn.BatchNorm2d(3) ,
			nn.Tanh()
		)

		return

	def forward(self , x):
		encode_x = self.encoder(x)
		decode_x = self.decoder(encode_x)
		return (encode_x , decode_x)